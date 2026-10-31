"""Configuration for the large-corpus RAG engine.

Every value can be overridden in backend/.env with a `RAG_` prefix (e.g. RAG_CHUNK_TOKENS=600), so these
settings never collide with the legacy Doc-AI settings in config.py. GROQ_API_KEY is also read unprefixed.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]     # backend/
load_dotenv(ROOT / ".env")


def _env(name: str, default):
    raw = os.getenv(f"RAG_{name}")
    if raw is None and name == "GROQ_API_KEY":
        raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return type(default)(raw)


# --- paths -------------------------------------------------------------------
DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
CHUNKS_DIR = INDEX_DIR / "chunks"          # per-document chunk JSONL (deterministic, reproducible)
EMB_DIR = INDEX_DIR / "embeddings"         # per-document precomputed embeddings (.npy)
MANIFEST_PATH = INDEX_DIR / "manifest.json"
LOG_DIR = DATA_DIR / "logs"
EVAL_DIR = DATA_DIR / "eval"

# --- extraction / OCR ----------------------------------------------------------
OCR_ENABLED = _env("OCR_ENABLED", True)
OCR_ENGINE = _env("OCR_ENGINE", "auto")          # auto | rapidocr | tesseract
OCR_DPI = _env("OCR_DPI", 200)
OCR_MIN_PAGE_CHARS = _env("OCR_MIN_PAGE_CHARS", 40)   # fewer native chars than this => treat page as scanned
OCR_EMBEDDED_IMAGES = _env("OCR_EMBEDDED_IMAGES", True)
OCR_IMAGE_MIN_AREA = _env("OCR_IMAGE_MIN_AREA", 0.20)  # min fraction of page area for an embedded image to be OCR'd
HEADER_FOOTER_BAND = _env("HEADER_FOOTER_BAND", 0.10)  # top/bottom fraction of page scanned for running headers
HEADER_FOOTER_MIN_FREQ = _env("HEADER_FOOTER_MIN_FREQ", 0.02)

# --- chunking -----------------------------------------------------------------
CHUNK_TOKENS = _env("CHUNK_TOKENS", 500)        # tokens measured with the embedding model's tokenizer
CHUNK_OVERLAP = _env("CHUNK_OVERLAP", 75)       # 15% overlap

# --- models (all free / open-source) ------------------------------------------
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PREFIX = _env("EMBED_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
EMBED_BATCH = _env("EMBED_BATCH", 16)
RERANK_ENABLED = _env("RERANK_ENABLED", True)
RERANK_MODEL = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_MAX_LEN = _env("RERANK_MAX_LEN", 256)       # query + chunk head; ~2x faster than 384 on CPU

# --- retrieval ----------------------------------------------------------------
TOP_K_DENSE = _env("TOP_K_DENSE", 30)
TOP_K_BM25 = _env("TOP_K_BM25", 30)
RRF_K = _env("RRF_K", 60)
RERANK_CANDIDATES = _env("RERANK_CANDIDATES", 12)
TOP_K_CONTEXT = _env("TOP_K_CONTEXT", 5)
HNSW_M = _env("HNSW_M", 32)
HNSW_EF_CONSTRUCTION = _env("HNSW_EF_CONSTRUCTION", 200)
HNSW_EF_SEARCH = _env("HNSW_EF_SEARCH", 100)

# --- generation ---------------------------------------------------------------
GROQ_API_KEY = _env("GROQ_API_KEY", "")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
# Measured on an i5-1135G7 (CPU only): 1.5B reads ~150-180 tok/s and writes ~20 tok/s; 3B ~90 / ~12 tok/s.
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_THREADS = _env("OLLAMA_THREADS", 0)          # 0 = Ollama default (pinning 3-4 threads measured no faster)
# auto => Groq when a key is configured, otherwise local Ollama
LLM_PROVIDER = _env("LLM_PROVIDER", "auto")
LLM_TEMPERATURE = _env("LLM_TEMPERATURE", 0.1)
# Local CPU models need a much smaller prompt and shorter answers to stay near the 2-5 s budget.
LLM_MAX_TOKENS_HOSTED = _env("LLM_MAX_TOKENS_HOSTED", 350)
LLM_MAX_TOKENS_LOCAL = _env("LLM_MAX_TOKENS_LOCAL", 90)
ANSWER_WORDS_HOSTED = _env("ANSWER_WORDS_HOSTED", 120)
ANSWER_WORDS_LOCAL = _env("ANSWER_WORDS_LOCAL", 45)
CONTEXT_CHUNKS_LOCAL = _env("CONTEXT_CHUNKS_LOCAL", 3)
CONTEXT_TOKENS_HOSTED = _env("CONTEXT_TOKENS_HOSTED", 2500)
CONTEXT_TOKENS_LOCAL = _env("CONTEXT_TOKENS_LOCAL", 260)

for _d in (PDF_DIR, CHROMA_DIR, CHUNKS_DIR, EMB_DIR, LOG_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
