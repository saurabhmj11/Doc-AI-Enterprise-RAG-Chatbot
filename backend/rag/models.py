"""Lazily-loaded open-source models (shared singletons)."""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from . import config

_lock = threading.Lock()


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    with _lock:
        model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    model.max_seq_length = 512
    return model


@lru_cache(maxsize=1)
def tokenizer():
    return embedder().tokenizer


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL, max_length=config.RERANK_MAX_LEN, device="cpu")


def count_tokens(texts: list[str]) -> list[int]:
    enc = tokenizer()(texts, add_special_tokens=False, truncation=False)["input_ids"]
    return [len(ids) for ids in enc]


def embed_passages(texts: list[str], batch_size: int | None = None, progress=None) -> np.ndarray:
    model = embedder()
    bs = batch_size or config.EMBED_BATCH
    out = []
    for i in range(0, len(texts), bs):
        out.append(model.encode(texts[i:i + bs], batch_size=bs, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False))
        if progress:
            progress(min(i + bs, len(texts)), len(texts))
    if not out:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return np.vstack(out).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return embedder().encode(config.EMBED_QUERY_PREFIX + text, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
