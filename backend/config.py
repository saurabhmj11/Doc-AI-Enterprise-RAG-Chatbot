import os

try:
    from pydantic_settings import BaseSettings  # type: ignore
except ImportError:
    try:
        from pydantic import BaseSettings  # type: ignore
    except ImportError:
        # Fallback for environments where pydantic-settings is not yet installed
        class BaseSettings:  # type: ignore
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                # Load values from environment variables if present
                for attr, default_val in self.__class__.__dict__.items():
                    if attr.startswith("_") or callable(default_val):
                        continue
                    env_val = os.getenv(attr.upper())
                    if env_val is not None:
                        if isinstance(default_val, bool):
                            setattr(self, attr, env_val.lower() in ("true", "1", "yes"))
                        elif isinstance(default_val, int):
                            setattr(self, attr, int(env_val))
                        elif isinstance(default_val, float):
                            setattr(self, attr, float(env_val))
                        elif isinstance(default_val, list):
                            setattr(self, attr, [x.strip() for x in env_val.split(",") if x.strip()])
                        else:
                            setattr(self, attr, env_val)
                    elif not hasattr(self, attr):
                        setattr(self, attr, default_val)

from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    gemini_api_key: str = ""
    gemini_model: str = "models/gemini-flash-latest"
    
    # LLM Configuration
    # 'online' (Gemini) or 'offline' (Ollama) or 'oss' (sentence-transformers only, no LLM)
    llm_mode: str = "online"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Document processing
    max_file_size_mb: int = 50  # Raised to support large PDFs (≥200 pages)
    allowed_extensions: list[str] = ["pdf", "docx", "txt"]
    
    # OCR settings
    enable_ocr: bool = True
    ocr_language: str = "eng"
    
    # Chunking settings (token-based for challenge requirements)
    chunk_size: int = 800       # Target: 500-1000 tokens per chunk
    chunk_overlap: int = 120    # ~15% overlap
    use_token_chunking: bool = True  # Token-based vs char-based chunking
    
    # Retrieval settings
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    similarity_threshold: float = 0.15
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.75
    low_confidence_threshold: float = 0.35
    grounding_coverage_threshold: float = 0.6
    
    # Embedding model (OSS by default for challenge compliance)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_mode: str = "oss"  # 'oss' (sentence-transformers) or 'gemini' or 'ollama'
    
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
    enable_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32

    # Hybrid Search
    enable_hybrid_search: bool = True
    hybrid_fusion_strategy: str = "rrf"    # "rrf" (Reciprocal Rank Fusion) or "alpha"
    hybrid_alpha: float = 0.5              # Weight for dense score when strategy="alpha"
    hybrid_rrf_k: int = 60                 # RRF constant — standard value per paper

    # Query Enhancement
    enable_hyde: bool = False
    enable_query_expansion: bool = True
    
    # MCP Server
    mcp_server_name: str = "ultra-doc-intelligence"
    mcp_server_version: str = "1.0.0"
    
    class Config:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
