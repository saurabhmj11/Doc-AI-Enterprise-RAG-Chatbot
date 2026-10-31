"""Hybrid retrieval: dense (HNSW in ChromaDB) + BM25, fused with Reciprocal Rank Fusion,
then refined by a cross-encoder reranker. Returns rich per-stage traces for visualisation."""
from __future__ import annotations

import time

import numpy as np

from . import config, store
from .models import embed_query, reranker


def retrieve(query: str, top_k: int | None = None, rerank: bool | None = None) -> dict:
    top_k = top_k or config.TOP_K_CONTEXT
    rerank = config.RERANK_ENABLED if rerank is None else rerank
    timings = {}

    t0 = time.perf_counter()
    qvec = embed_query(query)
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dense = store.dense_search(qvec, config.TOP_K_DENSE)
    timings["dense_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    lex = store.lexical().search(query, config.TOP_K_BM25)
    timings["bm25_ms"] = (time.perf_counter() - t0) * 1000

    # Reciprocal Rank Fusion
    t0 = time.perf_counter()
    cands: dict[str, dict] = {}
    for rank, (cid, score) in enumerate(dense, 1):
        cands.setdefault(cid, {"chunk_id": cid})
        cands[cid].update(dense_rank=rank, dense_score=round(score, 4))
    for rank, (cid, score) in enumerate(lex, 1):
        cands.setdefault(cid, {"chunk_id": cid})
        cands[cid].update(bm25_rank=rank, bm25_score=round(score, 3))
    for c in cands.values():
        c["rrf_score"] = sum(1.0 / (config.RRF_K + c[k]) for k in ("dense_rank", "bm25_rank") if k in c)
    fused = sorted(cands.values(), key=lambda c: -c["rrf_score"])
    for i, c in enumerate(fused, 1):
        c["rrf_rank"] = i
    timings["fuse_ms"] = (time.perf_counter() - t0) * 1000

    # skip ids the chunk store doesn't know yet (e.g. a document still being ingested)
    fused = [c for c in fused if store.get_chunk(c["chunk_id"])]
    pool = fused[: config.RERANK_CANDIDATES]
    for c in pool:
        ch = store.get_chunk(c["chunk_id"])
        c.update(text=ch["text"], filename=ch["filename"], doc_id=ch["doc_id"], pages=ch["pages"],
                 page_start=ch["page_start"], page_end=ch["page_end"], bboxes=ch["bboxes"],
                 source=ch["source"], n_tokens=ch["n_tokens"])

    t0 = time.perf_counter()
    if rerank and pool:
        scores = reranker().predict([(query, c["text"]) for c in pool], batch_size=len(pool),
                                    show_progress_bar=False)
        for c, s in zip(pool, np.asarray(scores, dtype=float)):
            c["rerank_score"] = round(float(s), 4)
        ranked = sorted(pool, key=lambda c: -c["rerank_score"])
    else:
        ranked = pool
    for i, c in enumerate(ranked, 1):
        c["final_rank"] = i
        c["selected"] = i <= top_k
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
    timings["retrieval_ms"] = sum(timings.values())

    return {
        "query": query,
        "candidates": ranked,                 # every reranked candidate, selected ones flagged
        "results": ranked[:top_k],
        "counts": {"dense": len(dense), "bm25": len(lex), "fused": len(fused), "reranked": len(pool)},
        "timings": {k: round(v, 1) for k, v in timings.items()},
    }
