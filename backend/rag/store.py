"""Persistence layer: ChromaDB (HNSW ANN index) + in-memory BM25 + chunk/manifest files."""
from __future__ import annotations

import json
import re
import threading
from collections import Counter

import numpy as np
from scipy import sparse

from . import config

COLLECTION = "pdf_chunks"

_STOP = set("""a an and are as at be but by for from has have he her his i if in into is it its of on or
that the their them then there these they this to was were which will with what when where who why how
does do did can could would should may might than so such not no yes you your we our""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def lex_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


# ----------------------------------------------------------------------------- vector DB
_client = None
_collection = None
_col_lock = threading.Lock()


def collection():
    global _client, _collection
    with _col_lock:
        if _collection is None:
            import chromadb
            from chromadb.config import Settings
            _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR),
                                                settings=Settings(anonymized_telemetry=False))
            _collection = _client.get_or_create_collection(
                COLLECTION,
                embedding_function=None,       # embeddings are precomputed by us
                configuration={"hnsw": {
                    "space": "cosine",
                    "max_neighbors": config.HNSW_M,
                    "ef_construction": config.HNSW_EF_CONSTRUCTION,
                    "ef_search": config.HNSW_EF_SEARCH,
                }},
            )
        return _collection


def _meta(c: dict) -> dict:
    return {
        "doc_id": c["doc_id"], "filename": c["filename"], "chunk_index": c["chunk_index"],
        "page_start": c["page_start"], "page_end": c["page_end"],
        "pages": ",".join(map(str, c["pages"])), "bboxes": json.dumps(c["bboxes"]),
        "source": c["source"], "lang": c["lang"], "n_tokens": c["n_tokens"],
    }


def upsert(chunks: list[dict], embeddings: np.ndarray, batch: int = 1000, progress=None) -> None:
    col = collection()
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.upsert(ids=[c["chunk_id"] for c in part],
                   embeddings=embeddings[i:i + batch].tolist(),
                   documents=[c["text"] for c in part],
                   metadatas=[_meta(c) for c in part])
        if progress:
            progress(min(i + batch, len(chunks)), len(chunks))


def delete_doc(doc_id: str) -> None:
    collection().delete(where={"doc_id": doc_id})


def dense_search(qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
    res = collection().query(query_embeddings=[qvec.tolist()], n_results=k, include=["distances"])
    return [(cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0])]


# ----------------------------------------------------------------------------- manifest + chunk files
def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"documents": {}}


def save_manifest(m: dict) -> None:
    tmp = config.MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
    tmp.replace(config.MANIFEST_PATH)


def save_doc_chunks(doc_id: str, chunks: list[dict], embeddings: np.ndarray) -> None:
    with open(config.CHUNKS_DIR / f"{doc_id}.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(config.EMB_DIR / f"{doc_id}.npy", embeddings)


def load_doc_chunks(doc_id: str) -> list[dict]:
    path = config.CHUNKS_DIR / f"{doc_id}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ----------------------------------------------------------------------------- BM25 + chunk lookup
class LexicalIndex:
    """Okapi BM25 over a sparse term matrix (fast vectorised scoring for ~10^4-10^5 chunks)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: dict[str, dict] = {}
        self.ids: list[str] = []
        self.vocab: dict[str, int] = {}
        self.matrix = None
        self.idf = None

    def build(self) -> None:
        manifest = load_manifest()
        chunks, ids = {}, []
        for doc_id in manifest["documents"]:
            for c in load_doc_chunks(doc_id):
                chunks[c["chunk_id"]] = c
                ids.append(c["chunk_id"])
        vocab: dict[str, int] = {}
        rows, cols, vals, lengths = [], [], [], []
        for r, cid in enumerate(ids):
            tf = Counter(lex_tokens(chunks[cid]["text"]))
            lengths.append(sum(tf.values()))
            for term, n in tf.items():
                rows.append(r)
                cols.append(vocab.setdefault(term, len(vocab)))
                vals.append(n)
        n_docs = max(len(ids), 1)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(ids), max(len(vocab), 1)), dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.float32) if lengths else np.ones(1, np.float32)
        avg = float(lengths.mean()) if len(ids) else 1.0
        df = np.bincount(cols, minlength=len(vocab)) if cols else np.zeros(1)
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # pre-compute BM25 term weights: tf*(k1+1) / (tf + k1*(1-b+b*len/avg))
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * lengths[tf.row] / avg)
        weights = tf.data * (self.k1 + 1) / denom
        self.matrix = sparse.csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)
        self.idf, self.vocab, self.ids, self.chunks = idf, vocab, ids, chunks

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = [self.vocab[t] for t in set(lex_tokens(query)) if t in self.vocab]
        if not terms or self.matrix is None:
            return []
        scores = self.matrix[:, terms] @ self.idf[terms]
        scores = np.asarray(scores).ravel()
        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


_lex: LexicalIndex | None = None
_lex_lock = threading.Lock()


def lexical() -> LexicalIndex:
    global _lex
    with _lex_lock:
        if _lex is None:
            _lex = LexicalIndex()
            _lex.build()
        return _lex


def refresh_lexical() -> None:
    global _lex
    idx = LexicalIndex()
    idx.build()
    with _lex_lock:
        _lex = idx


def get_chunk(chunk_id: str) -> dict | None:
    return lexical().chunks.get(chunk_id)
