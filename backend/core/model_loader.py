"""
Shared Model Loader
====================
Lazy loading for ML models to prevent blocking server startup.

Embedding modes:
  - 'oss'    : sentence-transformers (all-MiniLM-L6-v2) — FREE, local
  - 'gemini' : Google Gemini embedding API — requires API key
  - 'ollama' : Ollama embedding model — requires local Ollama server
"""

import logging
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Global singleton instance
_embedding_model = None


def get_embedding_model():
    """Get or create shared embedding model singleton with lazy imports."""
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    embedding_mode = settings.embedding_mode

    # ==========================================================
    # OSS MODE (sentence-transformers — FREE, no API key needed)
    # ==========================================================
    if embedding_mode == "oss":
        model_name = settings.embedding_model
        logger.info(f"Loading OSS embedding model: {model_name}")

        from sentence_transformers import SentenceTransformer
        import numpy as np

        class OSSEmbeddingModel:
            """Wrapper around sentence-transformers for consistent interface."""

            def __init__(self, model_name: str):
                self.model = SentenceTransformer(model_name)
                self.model_name = model_name

            def encode(self, texts, convert_to_numpy=True, task_type=None):
                """Encode texts into embeddings."""
                is_single = isinstance(texts, str)
                if is_single:
                    texts = [texts]

                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=convert_to_numpy,
                    show_progress_bar=False,
                    batch_size=64,
                )

                if is_single:
                    return embeddings[0]
                return embeddings

        _embedding_model = OSSEmbeddingModel(model_name)
        logger.info(f"OSS embedding model '{model_name}' loaded successfully")

    # ==========================================================
    # OLLAMA MODE
    # ==========================================================
    elif embedding_mode == "ollama" or settings.llm_mode == "offline":
        logger.info(f"Loading Ollama embedding model: {settings.ollama_embedding_model}")

        import ollama
        import numpy as np

        class OllamaEmbeddingModel:

            def __init__(self, model_name, base_url):
                self.model_name = model_name
                self.client = ollama.Client(host=base_url)

            def encode(self, texts, convert_to_numpy=True, task_type=None):
                is_single = isinstance(texts, str)
                if is_single:
                    texts = [texts]

                embeddings = []

                try:
                    resp = self.client.embeddings(
                        model=self.model_name,
                        input=texts
                    )

                    if isinstance(resp, dict):
                        if "embedding" in resp:
                            embeddings = [resp["embedding"]]
                        elif "embeddings" in resp:
                            embeddings = resp["embeddings"]
                        else:
                            raise RuntimeError("Unknown Ollama response format")
                    else:
                        embeddings = resp

                except Exception as e:
                    logger.error(f"Ollama embedding error: {e}")
                    raise RuntimeError(f"Ollama embedding failed: {e}")

                if is_single:
                    return np.array(embeddings[0]) if convert_to_numpy else embeddings[0]

                return np.array(embeddings) if convert_to_numpy else embeddings

        _embedding_model = OllamaEmbeddingModel(
            settings.ollama_embedding_model,
            settings.ollama_base_url
        )
        logger.info("Ollama embedding model loaded")

    # ==========================================================
    # GEMINI MODE
    # ==========================================================
    else:
        logger.info("Loading Gemini embedding model")

        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            genai = None

        if genai is None:
            raise RuntimeError(
                "google-generativeai package is not installed. "
                "Set embedding_mode='oss' in .env or backend/config.py to use free local embeddings."
            )

        class GeminiEmbeddingModel:

            def __init__(self, api_key):
                if not api_key:
                    raise ValueError("Gemini API Key is missing. Check backend/.env")
                genai.configure(api_key=api_key)
                self.model_name = "models/gemini-embedding-001"

            def encode(self, texts, convert_to_numpy=True, task_type="retrieval_document"):
                is_single = isinstance(texts, str)
                if is_single:
                    texts = [texts]

                embeddings = []
                batch_size = 20

                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]

                    try:
                        result = genai.embed_content(
                            model=self.model_name,
                            content=batch
                        )

                        batch_embeddings = result.get("embedding")
                        if not batch_embeddings:
                            raise RuntimeError("Empty embeddings from Gemini")

                        if isinstance(batch_embeddings[0], float):
                            if len(batch) == 1:
                                batch_embeddings = [batch_embeddings]
                            else:
                                raise RuntimeError("Unexpected embedding structure")

                        embeddings.extend(batch_embeddings)

                    except Exception as e:
                        logger.error(f"Gemini embedding batch error: {e}")
                        raise RuntimeError(f"Embedding generation failed: {e}")

                if is_single:
                    return np.array(embeddings[0]) if convert_to_numpy else embeddings[0]

                return np.array(embeddings) if convert_to_numpy else embeddings

        _embedding_model = GeminiEmbeddingModel(settings.gemini_api_key)
        logger.info("Gemini embedding model loaded")

    return _embedding_model
