"""
Ultra Doc-Intelligence API
Author: Development Team
Last Updated: Feb 2024

Quick project I put together for analyzing logistics docs.
Uses RAG + Gemini for Q&A, plus some basic extraction logic.
"""

import os
from dotenv import load_dotenv

# Load env vars from the same directory as this file
# This must happen BEFORE importing config or other modules that use settings
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware # Enable if SSL is handled at app level

from api.routes import router
from api.debug import router as debug_router
from api.auth import router as auth_router
from api.rag_routes import router as rag_router, warmup as rag_warmup
from config import get_settings
from core.error_handling import setup_logging

settings = get_settings()

# Configure logging before initializing components
setup_logging()

# spin up the app
app = FastAPI(
    title="Ultra Doc-Intelligence",
    description="Logistics document Q&A and data extraction tool",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.on_event("startup")
async def _startup_bm25_migration():
    """
    On server start, backfill BM25 sparse indexes for any documents already
    stored in ChromaDB from before the hybrid upgrade.  Runs in a background
    thread so it does not block the server from accepting requests.
    """
    import asyncio
    import threading
    import logging

    _log = logging.getLogger("startup.bm25_migration")

    def _run():
        try:
            from core.vector_store import get_vector_store
            from core.bm25_index import get_bm25_index

            vs = get_vector_store()
            bm25 = get_bm25_index()
            n = bm25.migrate_from_vector_store(vs)
            _log.info(f"BM25 startup migration finished: {n} document(s) backfilled")
        except Exception as exc:
            _log.error(f"BM25 startup migration error: {exc}", exc_info=True)

    # Run in a thread so the async event loop is not blocked
    threading.Thread(target=_run, daemon=True, name="bm25-migration").start()


@app.on_event("startup")
async def _startup_rag_warmup():
    """Pre-load the large-corpus RAG models/indexes so the first query meets the 2-5s budget."""
    rag_warmup()



# Security: Trusted Hosts
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["doc-ai-backend-bpez.onrender.com", "localhost", "127.0.0.1", "testserver"]
)

# CORS - Allow production domain + localhost for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://doc-ai-frontend.onrender.com",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# wire up the routes
app.include_router(router, prefix="/api", tags=["Documents"])
app.include_router(debug_router, prefix="/api/debug", tags=["Debug"])
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(rag_router, prefix="/api/rag", tags=["Large-corpus RAG"])


@app.get("/")
async def root():
    """Basic info about the API"""
    return {
        "app": "Ultra Doc-Intelligence",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Quick health check - useful for monitoring"""
    from core.error_handling import get_gemini_circuit_breaker, get_ollama_circuit_breaker
    
    # Get circuit breaker states
    gemini_cb = get_gemini_circuit_breaker() if settings.llm_mode == "online" else None
    ollama_cb = get_ollama_circuit_breaker() if settings.llm_mode == "offline" else None
    
    health_status = {
        "status": "ok",
        "llm_mode": settings.llm_mode,
        "llm_configured": bool(settings.gemini_api_key) if settings.llm_mode == "online" else True,
    }
    
    # Add circuit breaker status
    if gemini_cb:
        health_status["gemini_circuit_breaker"] = gemini_cb.state.value
    if ollama_cb:
        health_status["ollama_circuit_breaker"] = ollama_cb.state.value
    
    return health_status


# for running directly with python main.py
if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
