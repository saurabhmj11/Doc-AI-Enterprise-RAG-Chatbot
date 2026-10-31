"""Build a synthetic, reproducible evaluation set from the indexed corpus.

For N chunks sampled (fixed seed, stratified by document, prose-like chunks only) the LLM writes one
specific question answerable from that passage. Gold label = (document, pages) of the source chunk.

Usage: python scripts/build_eval.py [--n 40] [--seed 7]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import config, llm, store  # noqa: E402

PROMPT = """Below is a passage from the document "{filename}".
Write ONE specific question that a reader could answer using only this passage.
The question must be self-contained (mention the concrete topic, never say "the passage" or "the text"),
and must not be a yes/no question. Output only the question.

Passage:
{text}"""


def prose_like(text: str) -> bool:
    words = text.split()
    if len(words) < 200:
        return False
    alpha = sum(w.isalpha() for w in words) / len(words)
    return alpha > 0.72 and "...." not in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    docs = store.load_manifest()["documents"]
    per_doc = {d: [c for c in store.load_doc_chunks(d) if prose_like(c["text"])] for d in sorted(docs)}
    picks = []
    order = sorted(per_doc)
    while len(picks) < args.n and any(per_doc.values()):
        for d in order:
            if per_doc[d] and len(picks) < args.n:
                # skip front matter / back matter (first and last 5% of chunks)
                pool = per_doc[d]
                c = pool.pop(rng.randrange(len(pool)))
                total = docs[d]["chunks"]
                if 0.05 * total < c["chunk_index"] < 0.95 * total:
                    picks.append(c)

    items = []
    for i, c in enumerate(picks, 1):
        q = llm.complete([{"role": "user", "content": PROMPT.format(filename=c["filename"], text=c["text"][:2500])}],
                         max_tokens=80).strip().strip('"')
        q = re.sub(r"^(question:\s*)", "", q, flags=re.I).split("\n")[0].strip()
        items.append({"id": i, "question": q, "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                      "filename": c["filename"], "gold_pages": c["pages"]})
        print(f"[{i}/{len(picks)}] {c['filename']} p{c['pages']}: {q}", flush=True)

    out = config.EVAL_DIR / "eval_set.json"
    out.write_text(json.dumps({"generator": f"{llm.provider()}:{llm.model_name()}", "seed": args.seed,
                               "items": items}, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} questions to {out}")


if __name__ == "__main__":
    main()
