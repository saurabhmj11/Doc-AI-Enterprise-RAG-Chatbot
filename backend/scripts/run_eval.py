"""Evaluate the full RAG pipeline on the synthetic eval set.

Metrics
  * Retrieval: Recall@{1,3,5,10} and MRR@10 - a hit is a retrieved chunk from the gold document whose
    pages overlap the gold pages. Reported for dense-only, BM25-only, hybrid (RRF) and hybrid+rerank.
  * Latency: p50 / p95 of end-to-end, time-to-first-token and retrieval; share of queries within 5 s.
  * Citation accuracy: answer cites a source on the gold page; validity: cited [S#] tags exist.
  * Hallucination rate: answered questions where >1/3 of factual sentences are not lexically supported
    by their cited sources (cheap automatic proxy; see README for limitations).

Usage: python scripts/run_eval.py [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config, llm, store  # noqa: E402
from rag.models import embed_query  # noqa: E402
from rag.pipeline import answer  # noqa: E402

KS = (1, 3, 5, 10)


def is_hit(chunk: dict, item: dict) -> bool:
    return chunk["doc_id"] == item["doc_id"] and bool(set(chunk["pages"]) & set(item["gold_pages"]))


def first_hit(chunk_ids: list[str], item: dict) -> int | None:
    for rank, cid in enumerate(chunk_ids[:10], 1):
        c = store.get_chunk(cid)
        if c and is_hit(c, item):
            return rank
    return None


def summarize(ranks: list) -> dict:
    n = len(ranks)
    out = {f"recall@{k}": round(sum(r is not None and r <= k for r in ranks) / n, 3) for k in KS}
    out["mrr"] = round(sum(1 / r for r in ranks if r) / n, 3)
    return out


def p(vals, q):
    return round(float(np.percentile(vals, q)), 1) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    data = json.loads((config.EVAL_DIR / "eval_set.json").read_text(encoding="utf-8"))
    items = data["items"][: args.limit or None]

    # warm-up so model loading is not counted as query latency
    answer("warm up")

    ranks = {"dense": [], "bm25": [], "hybrid_rrf": [], "hybrid_rerank": []}
    rows = []
    for it in items:
        q = it["question"]
        ranks["dense"].append(first_hit([c for c, _ in store.dense_search(embed_query(q), 10)], it))
        ranks["bm25"].append(first_hit([c for c, _ in store.lexical().search(q, 10)], it))

        out = answer(q)
        cands = out["retrieval"]["candidates"]
        ranks["hybrid_rrf"].append(first_hit([c["chunk_id"] for c in sorted(cands, key=lambda c: c["rrf_rank"])], it))
        r = first_hit([c["chunk_id"] for c in cands], it)
        ranks["hybrid_rerank"].append(r)

        v = out["verification"]
        cited_gold = any(is_hit(c, it) for c in out["citations"])
        rows.append({
            "id": it["id"], "question": q, "filename": it["filename"], "gold_pages": it["gold_pages"], "rank": r,
            "total_ms": out["timings"]["total_ms"], "ttft_ms": out["timings"]["ttft_ms"],
            "retrieval_ms": out["timings"]["retrieval_ms"], "refused": v["refused"], "cited_gold": cited_gold,
            "citation_validity": v["citation_validity"], "grounding_score": v["grounding_score"],
            "hallucination_flag": v["hallucination_flag"], "answer": out["answer"],
            "citations": [f"{c['filename']} {c['page_label']}" for c in out["citations"]],
        })
        print(f"[{it['id']}] rank={r} total={out['timings']['total_ms']:.0f}ms ttft={out['timings']['ttft_ms']:.0f}ms "
              f"cited_gold={cited_gold} grounding={v['grounding_score']}", flush=True)

    answered = [r for r in rows if not r["refused"]]
    tot = [r["total_ms"] for r in rows]
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "n_questions": len(rows),
        "provider": llm.provider(), "model": llm.model_name(),
        "embed_model": config.EMBED_MODEL, "rerank_model": config.RERANK_MODEL,
        "retrieval": summarize(ranks["hybrid_rerank"]),
        "ablation": {k: summarize(v) for k, v in ranks.items()},
        "latency_ms": {
            "total": {"p50": p(tot, 50), "p95": p(tot, 95)},
            "ttft": {"p50": p([r["ttft_ms"] for r in rows], 50), "p95": p([r["ttft_ms"] for r in rows], 95)},
            "retrieval": {"p50": p([r["retrieval_ms"] for r in rows], 50), "p95": p([r["retrieval_ms"] for r in rows], 95)},
            "within_5s": round(sum(t <= 5000 for t in tot) / len(tot), 3),
        },
        "generation": {
            "answer_rate": round(len(answered) / len(rows), 3),
            "citation_accuracy": round(sum(r["cited_gold"] for r in rows) / len(rows), 3),
            "citation_validity": round(float(np.mean([r["citation_validity"] for r in answered])), 3) if answered else None,
            "hallucination_rate": round(sum(r["hallucination_flag"] for r in answered) / max(len(answered), 1), 3),
            "avg_grounding": round(float(np.mean([r["grounding_score"] for r in answered])), 3) if answered else None,
        },
        "items": rows,
    }
    (config.EVAL_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("retrieval", "ablation", "latency_ms", "generation")}, indent=2))


if __name__ == "__main__":
    main()
