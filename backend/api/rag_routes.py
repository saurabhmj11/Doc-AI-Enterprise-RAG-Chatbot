"""
Large-corpus RAG routes (mounted at /api/rag)

Chat (SSE streaming + JSON), retrieval-only search, live PDF ingestion jobs,
cited page previews, corpus status, live latency metrics and the offline eval report.
All routes require the same JWT as the rest of the API.
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pymupdf
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from core.auth import get_current_active_user, get_current_user
from rag import config, llm, store
from rag.ingest import STAGES, ingest_pdf
from rag.models import embedder, reranker
from rag.pipeline import answer, answer_stream
from rag.retrieve import retrieve

logger = logging.getLogger(__name__)
router = APIRouter()

_ready = {"ready": False, "warmup_s": None}
_jobs: dict[str, dict] = {}
_ingest_lock = threading.Lock()      # one ingestion at a time (CPU bound)


def warmup() -> None:
    """Load models, indexes and the local LLM in a background thread so the first query is fast."""
    def run():
        t0 = time.perf_counter()
        try:
            embedder(); store.lexical(); store.collection()
            if config.RERANK_ENABLED:
                reranker().predict([("warm", "up")], show_progress_bar=False)
            if store.lexical().ids:
                retrieve("warm up query")
            llm.warmup()
        except Exception:
            logger.exception("RAG warm-up failed")
        _ready.update(ready=True, warmup_s=round(time.perf_counter() - t0, 1))
        logger.info("RAG warm-up done in %ss", _ready["warmup_s"])
    threading.Thread(target=run, daemon=True, name="rag-warmup").start()


def _json(obj):
    return JSONResponse(json.loads(json.dumps(obj, default=float)))


# =====================================================
# CHAT
# =====================================================

class RagChatRequest(BaseModel):
    question: str
    history: list[dict] = []


def _sse(gen):
    for ev in gen:
        yield f"data: {json.dumps(ev, default=float)}\n\n"


@router.post("/chat")
def chat_stream(req: RagChatRequest, current_user: dict = Depends(get_current_active_user)):
    """Server-sent events: `retrieval` -> `token`* -> `done`."""
    if not req.question.strip():
        raise HTTPException(400, "empty question")
    return StreamingResponse(_sse(answer_stream(req.question.strip(), req.history)),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/ask")
def ask(req: RagChatRequest, current_user: dict = Depends(get_current_active_user)):
    """Same pipeline as /chat, returned as one JSON response."""
    out = answer(req.question.strip(), req.history)
    out.get("retrieval", {}).pop("candidates", None)
    return _json(out)


@router.get("/search")
def search(q: str, k: int = 10, current_user: dict = Depends(get_current_active_user)):
    """Retrieval only (no LLM)."""
    return _json(retrieve(q, top_k=k))


# =====================================================
# CORPUS / INGESTION
# =====================================================

@router.get("/status")
def status(current_user: dict = Depends(get_current_active_user)):
    docs = sorted(store.load_manifest()["documents"].values(), key=lambda d: d["filename"])
    return {
        **_ready,
        "provider": llm.provider(), "llm_model": llm.model_name(),
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL if config.RERANK_ENABLED else None,
        "vector_db": "ChromaDB (HNSW, cosine, M=%d, efC=%d, efS=%d)" % (
            config.HNSW_M, config.HNSW_EF_CONSTRUCTION, config.HNSW_EF_SEARCH),
        "chunking": {"tokens": config.CHUNK_TOKENS, "overlap": config.CHUNK_OVERLAP},
        "documents": docs,
        "totals": {
            "documents": len(docs),
            "pages": sum(d["pages"] for d in docs),
            "ocr_pages": sum(d["ocr_pages"] for d in docs),
            "ocr_images": sum(d["ocr_images"] for d in docs),
            "chunks": sum(d["chunks"] for d in docs),
            "vectors": store.collection().count(),
            "size_mb": round(sum(d["size_mb"] for d in docs), 1),
        },
    }


def _run_job(job_id: str, path: Path, force: bool):
    job = _jobs[job_id]

    def report(stage, done, total):
        job["stage"] = stage
        job["progress"][stage] = [done, total]
    with _ingest_lock:
        job["state"] = "running"
        try:
            job["result"] = ingest_pdf(path, force=force, report=report)
            job["state"] = "done"
        except Exception as e:
            logger.exception("ingestion failed")
            job.update(state="error", error=str(e))
    job["finished"] = time.time()


@router.post("/ingest")
async def ingest_upload(file: UploadFile, background: BackgroundTasks, force: bool = False,
                        current_user: dict = Depends(get_current_active_user)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are supported")
    dest = config.PDF_DIR / Path(file.filename).name
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"id": job_id, "filename": dest.name, "state": "queued", "stage": None,
                     "stages": STAGES, "progress": {}, "started": time.time()}
    background.add_task(_run_job, job_id, dest, force)
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def job_status(job_id: str, current_user: dict = Depends(get_current_active_user)):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


@router.get("/page/{doc_id}/{page}.png")
async def page_image(doc_id: str, page: int, token: str, hl: str | None = None, dpi: int = 110):
    """Render a PDF page; `hl=x0,y0,x1,y1` draws the cited chunk's bounding box.
    Auth via `?token=` because <img> tags cannot send an Authorization header."""
    await get_current_user(token)
    doc = store.load_manifest()["documents"].get(doc_id)
    if not doc:
        raise HTTPException(404, "unknown document")
    with pymupdf.open(config.PDF_DIR / doc["filename"]) as pdf:
        if not 1 <= page <= pdf.page_count:
            raise HTTPException(404, "page out of range")
        pg = pdf[page - 1]
        if hl:
            r = pymupdf.Rect(*map(float, hl.split(","))) + (-4, -4, 4, 4)
            pg.draw_rect(r, color=(0.95, 0.55, 0.0), fill=(1, 0.85, 0.3), fill_opacity=0.25, width=2)
        png = pg.get_pixmap(dpi=min(max(dpi, 50), 200)).tobytes("png")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


# =====================================================
# MONITORING / EVALUATION
# =====================================================

@router.get("/metrics")
def metrics(last: int = 200, current_user: dict = Depends(get_current_active_user)):
    path = config.LOG_DIR / "queries.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()[-last:]] if path.exists() else []
    if not rows:
        return {"count": 0, "recent": []}

    def pct(key, q):
        vals = [r["timings"][key] for r in rows if key in r["timings"]]
        return round(float(np.percentile(vals, q)), 1) if vals else None
    return {
        "count": len(rows),
        "latency_ms": {k: {"p50": pct(k, 50), "p95": pct(k, 95)}
                       for k in ("total_ms", "ttft_ms", "retrieval_ms", "rerank_ms", "generation_ms")},
        "within_5s": round(sum(r["timings"]["total_ms"] <= 5000 for r in rows) / len(rows), 3),
        "hallucination_flag_rate": round(sum(r["hallucination_flag"] for r in rows) / len(rows), 3),
        "avg_grounding": round(sum(r["grounding_score"] for r in rows) / len(rows), 3),
        "recent": rows[-15:][::-1],
    }


@router.get("/eval")
def eval_report(current_user: dict = Depends(get_current_active_user)):
    path = config.EVAL_DIR / "report.json"
    if not path.exists():
        return {"available": False}
    return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}
