"""Configuration and environment variables for Ultra Doc-Intelligence."""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    gemini_api_key: str = ""
    gemini_model: str = "models/gemini-flash-latest"
    
    # LLM Configuration
    llm_mode: str = "online"  # 'online' (Gemini) or 'offline' (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Document processing
    max_file_size_mb: int = 10
    allowed_extensions: list[str] = ["pdf", "docx", "txt"]
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval settings
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    similarity_threshold: float = 0.15  # Lowered for better retrieval
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.75
    low_confidence_threshold: float = 0.35
    grounding_coverage_threshold: float = 0.6
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    
    # API Error Handling
    retry_attempts: int = 3
    retry_base_delay: float = 2.0  # seconds
    circuit_breaker_threshold: int = 5  # failures before opening
    circuit_breaker_timeout: int = 60  # seconds before half-open
    enable_extractive_fallback: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_api_failures: bool = True
    log_file: str = "./logs/api_errors.log"
    
    # Reranking
    enable_reranking: bool = True           # Enabled as part of hybrid pipeline
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32

    # Hybrid Search
    enable_hybrid_search: bool = True
    hybrid_fusion_strategy: str = "rrf"    # "rrf" (Reciprocal Rank Fusion) or "alpha"
    hybrid_alpha: float = 0.5              # Weight for dense score when strategy="alpha"
    hybrid_rrf_k: int = 60                 # RRF constant — standard value per paper

    # Query Enhancement
    enable_hyde: bool = False              # HyDE: embed hypothetical answer for dense retrieval (costs 1 extra LLM call)
    enable_query_expansion: bool = True    # Expand BM25 query with logistics-domain synonyms (free, local)
    
    # MCP Server
    mcp_server_name: str = "ultra-doc-intelligence"
    mcp_server_version: str = "1.0.0"
    
    class Config:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"   # Allow SECRET_KEY, ADMIN_* etc. read by auth.py via os.getenv()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
