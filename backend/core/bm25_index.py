"""
BM25 Sparse Retrieval Index
============================
Provides keyword-based (sparse) retrieval to complement dense vector search.
Each document gets its own in-memory BM25 corpus built from its chunks.

Key design choices:
- Per-document BM25 objects so queries are always scoped to requested docs.
- Lightweight JSON persistence so indexes survive server restarts without
  re-embedding (we only need the raw chunk texts, not embeddings).
- Scores are normalized to [0, 1] so they can be fused with cosine scores.
"""

import re
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    In-memory BM25 index with per-document corpus management.

    Structure:
        _corpus[doc_id]    = list[list[str]]   tokenized chunk texts
        _chunk_meta[doc_id]= list[dict]         chunk metadata (id, page, text)
        _bm25[doc_id]      = BM25Okapi object   built from corpus

    Persistence: saves/loads via a single JSON file next to ChromaDB.
    """

    def __init__(self):
        self._corpus: dict[str, list[list[str]]] = {}
        self._chunk_meta: dict[str, list[dict]] = {}
        self._bm25: dict = {}          # doc_id → BM25Okapi (rebuilt on load)
        self._lock = threading.Lock()

        self._persist_path = Path(settings.chroma_persist_dir) / "bm25_corpus.json"
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, document_id: str, chunks) -> None:
        """
        Build a BM25 index for a new document from its chunks.

        Args:
            document_id: Unique document identifier.
            chunks: List of DocumentChunk dataclass objects.
        """
        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(c.text) for c in chunks]
        meta = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "page": c.page,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        with self._lock:
            self._corpus[document_id] = tokenized
            self._chunk_meta[document_id] = meta
            self._bm25[document_id] = BM25Okapi(tokenized)

        self._save()
        logger.info(f"BM25: indexed {len(chunks)} chunks for doc '{document_id}'")

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Run BM25 search over a single document's corpus.

        Returns:
            List of chunk dicts with an added 'bm25_score' field (normalized 0-1).
        """
        with self._lock:
            bm25 = self._bm25.get(document_id)
            meta = self._chunk_meta.get(document_id, [])

        if bm25 is None or not meta:
            logger.debug(f"BM25: no index for '{document_id}', returning empty")
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        raw_scores = bm25.get_scores(tokens)

        # Normalise to [0, 1] using max score
        max_score = float(max(raw_scores)) if max(raw_scores) > 0 else 1.0
        norm_scores = [float(s) / max_score for s in raw_scores]

        # Pair with metadata and sort descending
        scored = sorted(
            zip(norm_scores, meta),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in scored[:top_k]:
            results.append({
                **chunk,
                "bm25_score": score,
                "similarity_score": score,   # unified key for guardrails
            })

        return results

    def search_parallel(
        self,
        document_ids: list[str],
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Run BM25 search across multiple documents and merge by score.

        Returns top-k unique chunks across all documents.
        """
        import concurrent.futures

        futures_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for doc_id in document_ids:
                f = executor.submit(self.search, doc_id, query, top_k)
                futures_map[f] = doc_id

        all_results: list[dict] = []
        for f in concurrent.futures.as_completed(futures_map):
            try:
                all_results.extend(f.result())
            except Exception as exc:
                logger.error(f"BM25 search error for doc {futures_map[f]}: {exc}")

        # Deduplicate by chunk_id, keep highest score
        seen: dict[str, dict] = {}
        for r in all_results:
            cid = r["chunk_id"]
            if cid not in seen or r["bm25_score"] > seen[cid]["bm25_score"]:
                seen[cid] = r

        sorted_results = sorted(seen.values(), key=lambda x: x["bm25_score"], reverse=True)
        return sorted_results[:top_k]

    def document_exists(self, document_id: str) -> bool:
        """Return True if a BM25 index exists for this document."""
        return document_id in self._bm25

    def delete_document(self, document_id: str) -> bool:
        """Remove a document's BM25 index from memory and disk."""
        with self._lock:
            if document_id not in self._bm25:
                return False
            del self._corpus[document_id]
            del self._chunk_meta[document_id]
            del self._bm25[document_id]

        self._save()
        logger.info(f"BM25: deleted index for doc '{document_id}'")
        return True

    def migrate_from_vector_store(self, vector_store) -> int:
        """
        Backfill BM25 indexes for documents already in ChromaDB but not yet
        in the BM25 corpus.  Called once at server startup.

        Args:
            vector_store: The VectorStore singleton (has access to ChromaDB).

        Returns:
            Number of documents migrated.
        """
        from rank_bm25 import BM25Okapi

        migrated = 0
        all_doc_ids = list(vector_store._document_metadata.keys())

        for doc_id in all_doc_ids:
            if doc_id in self._bm25:
                continue  # already indexed

            try:
                # Pull chunk texts directly from ChromaDB
                collection = vector_store.client.get_collection("documents")
                result = collection.get(
                    where={"document_id": doc_id},
                    include=["documents", "metadatas"],
                )

                ids = result.get("ids", [])
                texts = result.get("documents", [])
                metas = result.get("metadatas", [])

                if not ids:
                    logger.warning(f"BM25 migration: no chunks found for doc '{doc_id}'")
                    continue

                tokenized = [_tokenize(t) for t in texts]
                meta = [
                    {
                        "chunk_id": ids[i],
                        "document_id": doc_id,
                        "text": texts[i],
                        "page": metas[i].get("page") if metas else None,
                        "chunk_index": metas[i].get("chunk_index", i) if metas else i,
                    }
                    for i in range(len(ids))
                ]

                with self._lock:
                    self._corpus[doc_id] = tokenized
                    self._chunk_meta[doc_id] = meta
                    self._bm25[doc_id] = BM25Okapi(tokenized)

                migrated += 1
                logger.info(
                    f"BM25 migration: indexed {len(ids)} chunks for existing doc '{doc_id}'"
                )

            except Exception as exc:
                logger.error(f"BM25 migration failed for doc '{doc_id}': {exc}")

        if migrated:
            self._save()  # Persist updated corpus
            logger.info(f"BM25 migration complete: {migrated} document(s) backfilled")
        else:
            logger.info("BM25 migration: all documents already indexed, nothing to do")

        return migrated

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist tokenized corpus and chunk metadata to JSON."""
        try:
            with self._lock:
                payload = {
                    "corpus": self._corpus,
                    "chunk_meta": self._chunk_meta,
                }

            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False))
            tmp.replace(self._persist_path)
        except Exception as exc:
            logger.warning(f"BM25: failed to persist index: {exc}")

    def _load(self) -> None:
        """Restore corpus from JSON and rebuild BM25 objects."""
        if not self._persist_path.exists():
            return

        try:
            from rank_bm25 import BM25Okapi

            payload = json.loads(self._persist_path.read_text())
            corpus: dict = payload.get("corpus", {})
            meta: dict = payload.get("chunk_meta", {})

            for doc_id, tokenized in corpus.items():
                self._corpus[doc_id] = tokenized
                self._chunk_meta[doc_id] = meta.get(doc_id, [])
                self._bm25[doc_id] = BM25Okapi(tokenized)

            logger.info(f"BM25: restored {len(corpus)} document indexes from disk")
        except Exception as exc:
            logger.warning(f"BM25: could not restore index from disk: {exc}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_bm25_index: Optional[BM25Index] = None
_bm25_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """Return the global BM25Index singleton (thread-safe)."""
    global _bm25_index
    with _bm25_lock:
        if _bm25_index is None:
            _bm25_index = BM25Index()
    return _bm25_index
