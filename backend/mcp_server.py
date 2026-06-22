"""
MCP Server for Ultra Doc-Intelligence

Exposes document intelligence capabilities via Model Context Protocol (MCP).
Allows AI assistants to upload documents, ask questions, and extract structured data.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource

from config import get_settings
from core.error_handling import setup_logging

settings = get_settings()
logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_document_processor = None
_vector_store = None
_rag_pipeline = None
_structured_extractor = None


def get_document_processor():
    """Lazy-load DocumentProcessor."""
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
    """Lazy-load BM25Index — keeps sparse index in sync with ChromaDB."""
    global _bm25_index
    if _bm25_index is None:
        from core.bm25_index import get_bm25_index as _get_bm25
        _bm25_index = _get_bm25()
    return _bm25_index

# Initialize server
app = Server(settings.mcp_server_name)


@app.list_resources()
async def list_resources() -> list[Resource]:
    """
    List all uploaded documents as resources.
    
    Returns:
        List of document resources with URIs like doc://document/{id}
    """
    try:
        vector_store = get_vector_store()
        
        # Get all document collections from ChromaDB
        # Note: This is a simplified version - you may need to maintain a separate document registry
        resources = []
        
        # For now, return a placeholder that shows the structure
        # In production, you'd query your database for all uploaded documents
        resources.append(
            Resource(
                uri="doc://documents",
                name="All Documents",
                mimeType="application/json",
                description="List of all uploaded logistics documents"
            )
        )
        
        logger.info(f"Listed {len(resources)} resources")
        return resources
        
    except Exception as e:
        logger.error(f"Error listing resources: {e}", exc_info=True)
        return []


@app.read_resource()
async def read_resource(uri: str) -> str:
    """
    Read a specific document resource.
    
    Args:
        uri: Resource URI (e.g., doc://document/abc123)
        
    Returns:
        Document metadata or content as JSON string
    """
    try:
        if uri == "doc://documents":
            # Return list of all documents
            # In production, query your database
            return '{"documents": [], "message": "No documents uploaded yet"}'
        
        # Parse document ID from URI
        if uri.startswith("doc://document/"):
            doc_id = uri.replace("doc://document/", "")
            
            # Get document metadata from vector store
            vector_store = get_vector_store()
            metadata = vector_store.get_document_metadata(doc_id)
            
            if metadata:
                import json
                return json.dumps(metadata, indent=2)
            else:
                return f'{{"error": "Document {doc_id} not found"}}'
        
        return f'{{"error": "Unknown resource URI: {uri}"}}'
        
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}", exc_info=True)
        return f'{{"error": "{str(e)}"}}'


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available tools for document operations.
    
    Returns:
        List of tool definitions
    """
    return [
        Tool(
            name="upload_document",
            description="Upload and process a logistics document (PDF, DOCX, or TXT)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the document file to upload"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="ask_question",
            description="Ask a question about one or more uploaded documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document IDs to query (get from upload_document or list_documents)"
                    },
                    "question": {
                        "type": "string",
                        "description": "Question to ask about the document(s)"
                    }
                },
                "required": ["document_ids", "question"]
            }
        ),
        Tool(
            name="extract_structured_data",
            description="Extract structured shipment data from a logistics document",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID to extract data from"
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="list_documents",
            description="List all uploaded documents",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="delete_document",
            description="Delete an uploaded document",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document ID to delete"
                    }
                },
                "required": ["document_id"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """
    Execute a tool based on its name.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        
    Returns:
        Tool execution results
    """
    try:
        if name == "upload_document":
            return await upload_document_tool(arguments)
        elif name == "ask_question":
            return await ask_question_tool(arguments)
        elif name == "extract_structured_data":
            return await extract_structured_data_tool(arguments)
        elif name == "list_documents":
            return await list_documents_tool(arguments)
        elif name == "delete_document":
            return await delete_document_tool(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def upload_document_tool(arguments: dict) -> Sequence[TextContent]:
    """Upload and process a document."""
    import uuid
    import json
    
    file_path = arguments.get("file_path")
    if not file_path:
        return [TextContent(type="text", text="Error: file_path is required")]
    
    file_path = Path(file_path)
    if not file_path.exists():
        return [TextContent(type="text", text=f"Error: File not found: {file_path}")]
    
    # Validate file type
    allowed_extensions = [".pdf", ".docx", ".txt"]
    if file_path.suffix.lower() not in allowed_extensions:
        return [TextContent(type="text", text=f"Error: File type {file_path.suffix} not supported. Use PDF, DOCX, or TXT")]
    
    try:
        # Get processor
        processor = get_document_processor()
        
        # Determine file type (remove dot)
        file_type = file_path.suffix[1:].lower()
        
        # Process document
        document_id, chunks = processor.process_file(str(file_path), file_type)
        
        # Store in vector database (dense retrieval)
        vector_store = get_vector_store()
        vector_store.add_document(
            document_id=document_id,
            chunks=chunks,
            filename=file_path.name,
            file_type=file_type,
            file_path=str(file_path)
        )

        # Index chunks in BM25 for hybrid sparse retrieval
        get_bm25_index().add_document(document_id, chunks)

        result = {
            "status": "success",
            "document_id": document_id,
            "filename": file_path.name,
            "num_chunks": len(chunks),
            "message": f"Document uploaded and processed successfully. Use document_id='{document_id}' for queries."
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Error in upload_document_tool: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error uploading document: {str(e)}")]


async def ask_question_tool(arguments: dict) -> Sequence[TextContent]:
    """Ask a question about documents."""
    import json
    
    document_ids = arguments.get("document_ids", [])
    question = arguments.get("question")
    
    if not document_ids:
        return [TextContent(type="text", text="Error: document_ids is required")]
    if not question:
        return [TextContent(type="text", text="Error: question is required")]
    
    try:
        rag = get_rag_pipeline()
        result = rag.ask(document_ids, question)
        
        # Format response
        response = {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "confidence_level": result["confidence_level"],
            "sources": result["sources"],
            "reranked": result.get("reranked", False)
        }
        
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error answering question: {str(e)}")]


async def extract_structured_data_tool(arguments: dict) -> Sequence[TextContent]:
    """Extract structured data from a document."""
    import json
    
    document_id = arguments.get("document_id")
    if not document_id:
        return [TextContent(type="text", text="Error: document_id is required")]
    
    try:
        # Get document text
        vector_store = get_vector_store()
        chunks = vector_store.get_all_chunks(document_id)
        
        if not chunks:
            return [TextContent(type="text", text=f"Error: Document {document_id} not found")]
        
        # Combine chunks to get full text
        full_text = "\n\n".join([chunk["text"] for chunk in chunks])
        
        # Extract structured data
        extractor = get_structured_extractor()
        shipment_data, confidence = extractor.extract(full_text)
        
        # Convert to dict
        result = {
            "document_id": document_id,
            "confidence": confidence,
            "data": shipment_data.dict()
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error extracting data: {str(e)}")]


async def list_documents_tool(arguments: dict) -> Sequence[TextContent]:
    """List all uploaded documents."""
    import json
    
    try:
        # In production, query your database for all documents
        # For now, return placeholder
        result = {
            "documents": [],
            "count": 0,
            "message": "Document listing requires a document registry. Upload documents using upload_document tool."
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error listing documents: {str(e)}")]


async def delete_document_tool(arguments: dict) -> Sequence[TextContent]:
    """Delete a document."""
    import json
    
    document_id = arguments.get("document_id")
    if not document_id:
        return [TextContent(type="text", text="Error: document_id is required")]
    
    try:
        vector_store = get_vector_store()
        vector_store.delete_document(document_id)

        # Also remove from BM25 sparse index
        get_bm25_index().delete_document(document_id)

        result = {
            "status": "success",
            "document_id": document_id,
            "message": f"Document {document_id} deleted successfully"
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error deleting document: {str(e)}")]


async def main():
    """Run the MCP server."""
    # Setup logging
    setup_logging()
    logger.info(f"Starting MCP server: {settings.mcp_server_name} v{settings.mcp_server_version}")
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
