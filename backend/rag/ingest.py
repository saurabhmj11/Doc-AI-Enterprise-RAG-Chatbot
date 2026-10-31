"""Ingestion pipeline: PDF -> extract/OCR -> clean -> chunk -> embed -> vector DB.

CLI:  python -m rag.ingest [paths or folders ...] [--force]
Idempotent: documents are keyed by content hash and skipped when already indexed.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from . import config, store
from .chunking import chunk_pages, doc_id_for
from .extract import extract_pdf
from .models import embed_passages

log = logging.getLogger(__name__)

STAGES = ["extract", "chunk", "embed", "index"]


def ingest_pdf(path: Path, force: bool = False, report=None, refresh: bool = True) -> dict:
    """Ingest one PDF. `report(stage, done, total)` receives live progress."""
    report = report or (lambda *a: None)
    path = Path(path)
    doc_id = doc_id_for(path)
    manifest = store.load_manifest()
    if doc_id in manifest["documents"] and not force:
        report("skipped", 1, 1)
        return manifest["documents"][doc_id]

    t = {}
    t0 = time.perf_counter()
    pages, stats = extract_pdf(path, progress=lambda d, n: report("extract", d, n))
    t["extract"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    report("chunk", 0, 1)
    chunks = [c.to_dict() for c in chunk_pages(pages, doc_id, path.name)]
    report("chunk", 1, 1)
    t["chunk"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    emb = embed_passages([c["text"] for c in chunks], progress=lambda d, n: report("embed", d, n))
    t["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    store.delete_doc(doc_id)
    store.upsert(chunks, emb, progress=lambda d, n: report("index", d, n))
    store.save_doc_chunks(doc_id, chunks, emb)
    t["index"] = time.perf_counter() - t0

    entry = {
        "doc_id": doc_id,
        "filename": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        **stats,
        "chunks": len(chunks),
        "avg_chunk_tokens": round(sum(c["n_tokens"] for c in chunks) / max(len(chunks), 1)),
        "embed_model": config.EMBED_MODEL,
        "embed_dim": int(emb.shape[1]) if len(emb) else 0,
        "timings_s": {k: round(v, 2) for k, v in t.items()},
        "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest = store.load_manifest()
    manifest["documents"][doc_id] = entry
    store.save_manifest(manifest)
    if refresh:
        store.refresh_lexical()
    report("done", 1, 1)
    return entry


def _collect(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out.extend(sorted(p.glob("*.pdf")) if p.is_dir() else [p])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=[str(config.PDF_DIR)])
    ap.add_argument("--force", action="store_true", help="re-ingest even if already indexed")
    args = ap.parse_args()

    files = _collect(args.paths)
    print(f"Ingesting {len(files)} PDF(s) with {config.EMBED_MODEL}")
    t_all = time.perf_counter()
    for i, f in enumerate(files, 1):
        last = {"stage": None}

        def report(stage, done, total, _f=f):
            if stage != last["stage"] or done == total or done % 50 == 0:
                print(f"  [{i}/{len(files)}] {_f.name[:45]:45s} {stage:8s} {done}/{total}", flush=True)
                last["stage"] = stage

        e = ingest_pdf(f, force=args.force, report=report, refresh=False)
        print(f"  -> {e['pages']} pages ({e['ocr_pages']} OCR, {e['ocr_images']} images OCR'd), "
              f"{e['chunks']} chunks, lang={e['lang']}, timings={e['timings_s']}", flush=True)
    print(f"Done in {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
