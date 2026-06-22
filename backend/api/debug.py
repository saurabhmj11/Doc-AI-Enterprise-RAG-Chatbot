"""
Debug and Monitoring Router

Endpoints for system introspection, model discovery, and deep health checks.
All endpoints require authentication — debug info must NOT be public.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends

from config import get_settings
from api.schemas import ModelsResponse, ModelInfo
from core.auth import require_role

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List LLM models (admin)",
    dependencies=[Depends(require_role("admin"))],
)
async def list_models():
    """
    List available LLM models for the current provider (Gemini or Ollama).
    Requires admin role.
    """
    models_list = []

    try:
        if settings.llm_mode == "online":
            if not settings.gemini_api_key:
                raise HTTPException(status_code=400, detail="Gemini API Key not configured")

            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            for m in genai.list_models():
                models_list.append(ModelInfo(
                    name=m.name,
                    methods=m.supported_generation_methods
                ))
        else:
            import ollama
            client = ollama.Client(host=settings.ollama_base_url)
            ollama_models = client.list()

            if isinstance(ollama_models, dict) and "models" in ollama_models:
                for m in ollama_models["models"]:
                    models_list.append(ModelInfo(
                        name=m.get("name", "unknown"),
                        methods=["generateContent"],
                    ))
            elif isinstance(ollama_models, list):
                for m in ollama_models:
                    models_list.append(ModelInfo(
                        name=getattr(m, "name", str(m)),
                        methods=["generateContent"],
                    ))

        return ModelsResponse(models=models_list)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error listing models: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {exc}")


@router.get(
    "/health/extended",
    summary="Deep health check (admin)",
    dependencies=[Depends(require_role("admin"))],
)
async def extended_health():
    """
    Deep health check covering vector store, BM25 index, hybrid pipeline,
    reranker, and LLM connectivity.  Requires admin role.
    """
    from core.vector_store import get_vector_store
    from core.bm25_index import get_bm25_index

    vs = get_vector_store()
    bm25 = get_bm25_index()

    # BM25 index statistics
    bm25_indexed_docs = len(bm25._bm25)
    bm25_total_chunks = sum(len(c) for c in bm25._corpus.values())

    # Vector store document count
    vs_doc_count = len(vs._document_metadata)

    # Determine migration status
    unindexed = [
        doc_id for doc_id in vs._document_metadata
        if doc_id not in bm25._bm25
    ]

    return {
        "status": "ok",
        "vector_store": {
            "status": "connected",
            "documents": vs_doc_count,
        },
        "bm25_index": {
            "status": "active" if bm25_indexed_docs > 0 else "empty",
            "indexed_documents": bm25_indexed_docs,
            "total_chunks_indexed": bm25_total_chunks,
            "unindexed_documents": len(unindexed),  # should be 0 after migration
            "corpus_file": str(bm25._corpus_path),
        },
        "hybrid_pipeline": {
            "enabled": settings.enable_hybrid_search,
            "fusion_strategy": settings.hybrid_fusion_strategy,
            "rrf_k": settings.hybrid_rrf_k,
            "query_expansion": settings.enable_query_expansion,
            "hyde": settings.enable_hyde,
            "reranking": settings.enable_reranking,
            "reranker_model": settings.reranker_model if settings.enable_reranking else None,
        },
        "llm": {
            "mode": settings.llm_mode,
            "model": settings.gemini_model if settings.llm_mode == "online" else settings.ollama_model,
            "api_key_set": bool(settings.gemini_api_key),
        },
    }
