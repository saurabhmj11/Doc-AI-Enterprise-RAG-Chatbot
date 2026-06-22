"""
API Routes Module

FastAPI endpoints for document upload, Q&A, and structured extraction.
"""

import os
import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Optional
from core.auth import get_current_active_user

from config import get_settings
from api.schemas import (
    DocumentUploadResponse,
    AskRequest,
    AskResponse,
    SourceChunk,
    ExtractRequest,
    ExtractResponse,
    ErrorResponse,
    ConfigResponse,
    ConfigUpdate
)

settings = get_settings()
router = APIRouter()
logger = logging.getLogger(__name__)

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Lazy-loaded singletons to prevent blocking server startup
_document_processor = None
_vector_store = None
_rag_pipeline = None
_structured_extractor = None


def get_document_processor():
    """Lazy-load DocumentProcessor to avoid import-time heavy loading."""
    global _document_processor
    if _document_processor is None:
        from core.document_processor import DocumentProcessor
        _document_processor = DocumentProcessor()
    return _document_processor


def get_vector_store():
    """Lazy-load VectorStore."""
    global _vector_store
    if _vector_store is None:
        from core.vector_store import get_vector_store as _get_vs
        _vector_store = _get_vs()
    return _vector_store


def get_rag_pipeline():
    """Lazy-load RAGPipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        from core.rag_pipeline import get_rag_pipeline as _get_rag
        _rag_pipeline = _get_rag()
    return _rag_pipeline


def get_structured_extractor():
    """Lazy-load StructuredExtractor."""
    global _structured_extractor
    if _structured_extractor is None:
        from core.structured_extractor import get_structured_extractor as _get_ext
        _structured_extractor = _get_ext()
    return _structured_extractor


_bm25_index = None

def get_bm25_index():
    """Lazy-load BM25Index — keeps sparse index in sync with vector store."""
    global _bm25_index
    if _bm25_index is None:
        from core.bm25_index import get_bm25_index as _get_bm25
        _bm25_index = _get_bm25()
    return _bm25_index


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Upload and process a logistics document.
    
    Accepts PDF, DOCX, or TXT files.
    Returns document ID for subsequent queries.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Get file extension
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {settings.allowed_extensions}"
        )
    
    # Check file size
    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Save file to a UUID-based temp path — never use user-supplied filenames
    # on disk to prevent path traversal attacks (e.g. ../../main.py)
    safe_filename = f"{uuid.uuid4().hex}.{file_ext}"
    temp_path = UPLOAD_DIR / safe_filename
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process document
        document_id, chunks = get_document_processor().process_file(
            str(temp_path),
            file_ext
        )
        
        # Store in vector database
        vector_store = get_vector_store()
        vector_store.add_document(
            document_id=document_id,
            chunks=chunks,
            filename=file.filename,
            file_type=file_ext,
            file_path=str(temp_path)
        )

        # Index chunks in BM25 for hybrid sparse retrieval
        get_bm25_index().add_document(document_id, chunks)

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_type=file_ext,
            chunks_created=len(chunks),
            status="processed",
            message=f"Document processed successfully. {len(chunks)} chunks created."
        )
        
    except Exception as e:
        # Clean up on error
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Ask a question about one or more uploaded documents.
    
    Returns answer with confidence score, sources, and guardrail status.
    """
    # Determine document IDs to search
    doc_ids = request.document_ids
    if not doc_ids and request.document_id:
        doc_ids = [request.document_id]
        
    if not doc_ids:
        raise HTTPException(status_code=400, detail="No document provided. Please upload a document first.")

    # Validate documents exist
    vector_store = get_vector_store()
    for did in doc_ids:
        if not vector_store.document_exists(did):
            raise HTTPException(
                status_code=404,
                detail=f"Document {did} not found. Please upload a document first."
            )
    
    try:
        # Get answer from RAG pipeline (multi-doc supported)
        rag = get_rag_pipeline()
        result = rag.ask(
            document_ids=doc_ids,
            question=request.question
        )
        
        # Format sources
        sources = [
            SourceChunk(
                text=s["text"],
                filename=s.get("filename"),
                page=s.get("page"),
                chunk_id=s["chunk_id"],
                similarity_score=s["similarity_score"]
            )
            for s in result.get("sources", [])
        ]
        
        return AskResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            confidence_level=result["confidence_level"],
            sources=sources,
            guardrail_status=result["guardrail_status"],
            guardrail_message=result.get("guardrail_message")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/extract", response_model=ExtractResponse)
async def extract_structured_data(
    request: ExtractRequest,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Extract structured shipment data from an uploaded document.
    
    Returns JSON with shipment details and nulls for missing fields.
    """
    # Validate document exists
    vector_store = get_vector_store()
    if not vector_store.document_exists(request.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document {request.document_id} not found. Please upload a document first."
        )
    
    try:
        # Get document metadata
        metadata = vector_store.get_document_metadata(request.document_id)
        file_path = metadata["file_path"]
        file_type = metadata["file_type"]
        
        # Get full document text
        full_text = get_document_processor().get_full_text(file_path, file_type)
        
        # Extract structured data
        extractor = get_structured_extractor()
        shipment_data, confidence = extractor.extract(full_text)
        
        # Count non-null fields
        fields = [
            shipment_data.shipment_id,
            shipment_data.shipper,
            shipment_data.consignee,
            shipment_data.pickup_datetime,
            shipment_data.delivery_datetime,
            shipment_data.equipment_type,
            shipment_data.mode,
            shipment_data.rate,
            shipment_data.currency,
            shipment_data.weight,
            shipment_data.carrier_name
        ]
        fields_found = sum(1 for f in fields if f is not None)
        
        return ExtractResponse(
            document_id=request.document_id,
            extraction=shipment_data,
            extraction_confidence=round(confidence, 3),
            fields_found=fields_found,
            total_fields=11
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting data: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document_info(
    document_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get information about an uploaded document."""
    vector_store = get_vector_store()
    
    if not vector_store.document_exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    
    metadata = vector_store.get_document_metadata(document_id)
    return {
        "document_id": document_id,
        "filename": metadata["filename"],
        "file_type": metadata["file_type"],
        "chunks": metadata["chunk_count"]
    }


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Delete an uploaded document."""
    vector_store = get_vector_store()
    
    if not vector_store.document_exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get file path before deleting from vector store
    metadata = vector_store.get_document_metadata(document_id)
    file_path = Path(metadata["file_path"])
    
    # Delete from vector store
    vector_store.delete_document(document_id)

    # Delete from BM25 sparse index
    get_bm25_index().delete_document(document_id)

    # Delete file
    if file_path.exists():
        file_path.unlink()
    
    return {"status": "deleted", "document_id": document_id}


@router.get("/config", response_model=ConfigResponse)
async def get_config(current_user: dict = Depends(get_current_active_user)):
    """Get current configuration. Requires authentication."""
    current_settings = get_settings()
    
    return ConfigResponse(
        llm_mode=current_settings.llm_mode,
        gemini_api_key_configured=bool(current_settings.gemini_api_key),
        ollama_base_url=current_settings.ollama_base_url,
        ollama_model=current_settings.ollama_model
    )


@router.post("/config", response_model=ConfigResponse)
async def update_config(config_update: ConfigUpdate, current_user: dict = Depends(get_current_active_user)):
    """Update configuration and .env file. Requires authentication."""
    # 1. Update .env file
    env_path = Path(__file__).parent.parent / ".env"
    
    # Read existing env
    env_content = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    env_content[key] = val
    
    # Update values
    env_content["LLM_MODE"] = config_update.llm_mode
    if config_update.gemini_api_key:
        env_content["GEMINI_API_KEY"] = config_update.gemini_api_key
    if config_update.ollama_base_url:
        env_content["OLLAMA_BASE_URL"] = config_update.ollama_base_url
    if config_update.ollama_model:
        env_content["OLLAMA_MODEL"] = config_update.ollama_model
        
    # Write back to .env if not in immutable mode
    if os.getenv("IMMUTABLE_CONFIG", "false").lower() != "true":
        with open(env_path, "w") as f:
            for key, val in env_content.items():
                f.write(f"{key}={val}\n")
    else:
        logger.warning("Attempted to update config in IMMUTABLE_CONFIG mode. Changes will not persist.")
            
    # 2. Force reload settings
    # We need to clear lru_cache of get_settings
    from config import get_settings
    get_settings.cache_clear()
    
    # Also reload dotenv to pick up changes in process environment
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)
    
    # 3. Reload components that depend on settings
    # This is tricky without restart, but for now we'll just clear the singletons
    # so they get re-initialized on next request
    global _rag_pipeline, _structured_extractor, _document_processor, _vector_store
    
    # Pipeline and Extractor definitely need reload if LLM mode changes
    _rag_pipeline = None
    _structured_extractor = None
    
    # Vector store might need reload if embedding model changes (but that's harder to swap on fly)
    # We'll leave vector store for now as changing embedding model usually requires re-indexing
    
    # Get new settings verify
    new_settings = get_settings()
    
    return ConfigResponse(
        llm_mode=new_settings.llm_mode,
        gemini_api_key_configured=bool(new_settings.gemini_api_key),
        ollama_base_url=new_settings.ollama_base_url,
        ollama_model=new_settings.ollama_model
    )
