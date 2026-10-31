"""Deterministic token-based chunking with overlap and per-chunk provenance.

Rules (fixed, so re-running ingestion yields identical chunks):
  1. Pages -> blocks (paragraphs, reading order) -> sentences ("units").
  2. Units are packed greedily until CHUNK_TOKENS (embedding-tokenizer tokens) would be exceeded.
  3. The next chunk starts with the trailing units of the previous one totalling >= CHUNK_OVERLAP
     tokens (sentence-aligned overlap, ~15%).
  4. Units longer than CHUNK_TOKENS are split on word boundaries.
  5. A chunk may span a page break; it records every page it touches plus a bbox per page.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from . import config
from .extract import Page
from .models import count_tokens

_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\[\"'])")


@dataclass
class Unit:
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    source: str
    tokens: int = 0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    n_tokens: int
    page_start: int
    page_end: int
    pages: list[int]
    bboxes: dict          # {page: [x0, y0, x1, y1]} union of the chunk's units on that page
    source: str           # native | ocr | mixed
    lang: str

    def to_dict(self) -> dict:
        return asdict(self)


def doc_id_for(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def _units(pages: list[Page]) -> list[Unit]:
    units = []
    for p in pages:
        for b in p.blocks:
            for sent in _SENT_SPLIT.split(b.text):
                sent = sent.strip()
                if sent:
                    units.append(Unit(sent, p.number, b.bbox, b.source))
    counts = count_tokens([u.text for u in units]) if units else []
    for u, n in zip(units, counts):
        u.tokens = n
    # split oversize units on word boundaries
    out = []
    for u in units:
        if u.tokens <= config.CHUNK_TOKENS:
            out.append(u)
            continue
        words = u.text.split()
        per = max(1, int(len(words) * config.CHUNK_TOKENS * 0.9 / u.tokens))
        for i in range(0, len(words), per):
            piece = " ".join(words[i:i + per])
            out.append(Unit(piece, u.page, u.bbox, u.source, count_tokens([piece])[0]))
    return out


def _make_chunk(doc_id, filename, idx, units: list[Unit], page_lang: dict) -> Chunk:
    pages = sorted({u.page for u in units})
    bboxes = {}
    for u in units:
        b = bboxes.get(u.page)
        bboxes[u.page] = list(u.bbox) if b is None else [min(b[0], u.bbox[0]), min(b[1], u.bbox[1]),
                                                       max(b[2], u.bbox[2]), max(b[3], u.bbox[3])]
    sources = {u.source for u in units}
    return Chunk(
        chunk_id=f"{doc_id}:{idx:05d}",
        doc_id=doc_id,
        filename=filename,
        chunk_index=idx,
        text=" ".join(u.text for u in units),
        n_tokens=sum(u.tokens for u in units),
        page_start=pages[0],
        page_end=pages[-1],
        pages=pages,
        bboxes={str(k): [round(x, 1) for x in v] for k, v in bboxes.items()},
        source=sources.pop() if len(sources) == 1 else "mixed",
        lang=page_lang.get(pages[0], "unknown"),
    )


def chunk_pages(pages: list[Page], doc_id: str, filename: str) -> list[Chunk]:
    units = _units(pages)
    page_lang = {p.number: p.lang for p in pages}
    chunks: list[Chunk] = []
    cur: list[Unit] = []
    cur_tokens = 0
    fresh = 0      # units in `cur` that are not overlap carried from the previous chunk
    for u in units:
        if cur and cur_tokens + u.tokens > config.CHUNK_TOKENS:
            chunks.append(_make_chunk(doc_id, filename, len(chunks), cur, page_lang))
            fresh = 0
            # sentence-aligned overlap
            carry, carried = [], 0
            for prev in reversed(cur):
                if carried >= config.CHUNK_OVERLAP:
                    break
                carry.insert(0, prev)
                carried += prev.tokens
            if carried + u.tokens > config.CHUNK_TOKENS:
                carry, carried = [], 0
            cur, cur_tokens = carry, carried
        cur.append(u)
        cur_tokens += u.tokens
        fresh += 1
    if cur and fresh:
        chunks.append(_make_chunk(doc_id, filename, len(chunks), cur, page_lang))
    return chunks
