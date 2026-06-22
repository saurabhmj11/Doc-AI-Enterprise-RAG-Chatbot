"""
Hybrid Fuser
=============
Merges dense (bi-encoder) and sparse (BM25) retrieval result lists into a
single ranked list using one of two strategies:

  - "rrf"   : Reciprocal Rank Fusion — rank-based, no score calibration needed.
              Recommended for production.
              score(d) = Σ 1 / (k + rank_i(d))   where k is typically 60.

  - "alpha" : Weighted score blend.
              score(d) = α * dense_score + (1-α) * bm25_score
              Requires scores to be on comparable scales (both normalised 0-1).

References:
    Cormack, Clarke & Buettcher (2009) — Reciprocal Rank Fusion outperforms
    Condorcet and individual rank learning methods.
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
"""

import logging
from typing import Optional

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class HybridFuser:
    """
    Fuse dense and sparse retrieval results into a single ranked list.

    Args:
        strategy : "rrf" or "alpha"  (read from settings by default)
        alpha    : dense weight when strategy="alpha"  (0.0 – 1.0)
        rrf_k    : RRF smoothing constant (default: 60 per paper)
    """

    def __init__(
        self,
        strategy: Optional[str] = None,
        alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
    ):
        self.strategy = strategy or settings.hybrid_fusion_strategy
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.rrf_k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k

        if self.strategy not in ("rrf", "alpha"):
            logger.warning(
                f"HybridFuser: unknown strategy '{self.strategy}', falling back to 'rrf'"
            )
            self.strategy = "rrf"

        logger.info(
            f"HybridFuser initialised — strategy={self.strategy}, "
            f"alpha={self.alpha}, rrf_k={self.rrf_k}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Merge two ranked result lists into a single fused ranking.

        Args:
            dense_results  : Chunks from ChromaDB (must have 'chunk_id' key).
            sparse_results : Chunks from BM25     (must have 'chunk_id' key).
            top_k          : Maximum results to return.

        Returns:
            Merged list of chunk dicts sorted by fused score (descending).
            Each chunk gets an extra 'hybrid_score' field.
        """
        if not dense_results and not sparse_results:
            return []

        if self.strategy == "rrf":
            return self._rrf(dense_results, sparse_results, top_k)
        else:
            return self._alpha_blend(dense_results, sparse_results, top_k)

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf(
        self,
        dense: list[dict],
        sparse: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        RRF score for each chunk = Σ 1 / (k + rank)
        where rank is 1-indexed position in each list.
        """
        k = self.rrf_k
        scores: dict[str, float] = {}

        # Build a quick lookup: chunk_id → chunk dict (from either list)
        chunk_by_id: dict[str, dict] = {}

        for rank, chunk in enumerate(dense, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = chunk

        for rank, chunk in enumerate(sparse, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = chunk

        # Sort by fused RRF score
        sorted_ids = sorted(scores, key=lambda c: scores[c], reverse=True)

        results = []
        for cid in sorted_ids[:top_k]:
            merged = dict(chunk_by_id[cid])
            merged["hybrid_score"] = round(scores[cid], 6)
            merged["similarity_score"] = merged["hybrid_score"]  # normalise key
            results.append(merged)

        logger.debug(
            f"RRF fusion: {len(dense)} dense + {len(sparse)} sparse "
            f"→ {len(results)} unique chunks (top {top_k})"
        )
        return results

    # ------------------------------------------------------------------
    # Alpha-Blend Fusion
    # ------------------------------------------------------------------

    def _alpha_blend(
        self,
        dense: list[dict],
        sparse: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Alpha-blend score = α * dense_score + (1-α) * bm25_score.
        Both scores must be normalised to [0, 1].
        """
        alpha = self.alpha
        chunk_by_id: dict[str, dict] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}

        for chunk in dense:
            cid = chunk["chunk_id"]
            dense_scores[cid] = chunk.get("similarity_score", 0.0)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = chunk

        for chunk in sparse:
            cid = chunk["chunk_id"]
            sparse_scores[cid] = chunk.get("bm25_score", 0.0)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = chunk

        all_ids = set(dense_scores) | set(sparse_scores)
        blended: dict[str, float] = {}

        for cid in all_ids:
            d = dense_scores.get(cid, 0.0)
            s = sparse_scores.get(cid, 0.0)
            blended[cid] = alpha * d + (1.0 - alpha) * s

        sorted_ids = sorted(blended, key=lambda c: blended[c], reverse=True)

        results = []
        for cid in sorted_ids[:top_k]:
            merged = dict(chunk_by_id[cid])
            merged["hybrid_score"] = round(blended[cid], 6)
            merged["similarity_score"] = merged["hybrid_score"]
            results.append(merged)

        logger.debug(
            f"Alpha-blend fusion (α={alpha}): {len(dense)} dense + {len(sparse)} sparse "
            f"→ {len(results)} unique chunks (top {top_k})"
        )
        return results


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_hybrid_fuser: Optional[HybridFuser] = None


def get_hybrid_fuser() -> HybridFuser:
    """Return the global HybridFuser singleton."""
    global _hybrid_fuser
    if _hybrid_fuser is None:
        _hybrid_fuser = HybridFuser()
    return _hybrid_fuser
