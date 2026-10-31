"""PDF -> clean page text.

* Native text via PyMuPDF (block level, with bounding boxes, in reading order).
* OCR for scanned pages (little/no native text) and for large embedded images
  (RapidOCR / ONNX by default, Tesseract when installed and selected).
* Cleaning: unicode normalisation, de-hyphenation, running header/footer and
  page-number removal, whitespace normalisation.
* Language detection per page (langdetect, seeded => deterministic).
"""
from __future__ import annotations

import io
import logging
import re
import shutil
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import pymupdf as fitz

from . import config

log = logging.getLogger(__name__)


@dataclass
class Block:
    text: str
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 in PDF points
    source: str = "native"                     # native | ocr


@dataclass
class Page:
    number: int                                # 1-based
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)
    method: str = "native"                     # native | ocr | native+ocr
    lang: str = "unknown"

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


# ----------------------------------------------------------------------------- OCR
class OCR:
    """Lazy wrapper choosing RapidOCR (pure pip, ONNX) or Tesseract."""

    def __init__(self) -> None:
        self._engine = None
        self.name = None

    def _load(self):
        if self._engine is not None:
            return
        want = config.OCR_ENGINE
        if want in ("auto", "tesseract") and shutil.which("tesseract"):
            import pytesseract
            self._engine, self.name = pytesseract, "tesseract"
        elif want in ("auto", "rapidocr"):
            from rapidocr_onnxruntime import RapidOCR
            self._engine, self.name = RapidOCR(), "rapidocr"
        else:
            raise RuntimeError(f"OCR engine '{want}' unavailable")
        log.info("OCR engine: %s", self.name)

    def image_to_lines(self, png: bytes) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Return (text, bbox-in-pixels) lines."""
        self._load()
        if self.name == "tesseract":
            from PIL import Image
            img = Image.open(io.BytesIO(png))
            data = self._engine.image_to_data(img, output_type=self._engine.Output.DICT)
            lines: dict[tuple, list] = {}
            for i, word in enumerate(data["text"]):
                if word.strip() and float(data["conf"][i]) > 30:
                    key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    lines.setdefault(key, []).append((word, (x, y, x + w, y + h)))
            out = []
            for words in lines.values():
                xs0, ys0, xs1, ys1 = zip(*(b for _, b in words))
                out.append((" ".join(w for w, _ in words), (min(xs0), min(ys0), max(xs1), max(ys1))))
            return out
        result, _ = self._engine(png)
        out = []
        for box, text, score in result or []:
            if float(score) < 0.5 or not text.strip():
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append((_respace(text), (min(xs), min(ys), max(xs), max(ys))))
        return out


_GLUED = re.compile(r"[A-Za-z]{14,}")


def _respace(text: str) -> str:
    """RapidOCR's multilingual recogniser often drops spaces between English words
    ('Tofacilitatethe...'); re-segment long letter runs with a unigram language model."""
    import wordninja
    return _GLUED.sub(lambda m: " ".join(wordninja.split(m.group(0))), text)


_ocr = OCR()


def _ocr_region(page: fitz.Page, clip: fitz.Rect | None = None) -> list[Block]:
    """OCR a page (or a clipped region) and return blocks in PDF coordinates, grouped into paragraphs."""
    zoom = config.OCR_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, colorspace=fitz.csGRAY)
    lines = _ocr.image_to_lines(pix.tobytes("png"))
    if not lines:
        return []
    ox, oy = (clip.x0, clip.y0) if clip else (0, 0)
    lines = [(t, (b[0] / zoom + ox, b[1] / zoom + oy, b[2] / zoom + ox, b[3] / zoom + oy)) for t, b in lines]
    lines.sort(key=lambda l: (round(l[1][1] / 4), l[1][0]))
    # merge lines into paragraphs when the vertical gap is small
    blocks: list[Block] = []
    cur_text, cur_box = [], None
    for text, box in lines:
        if cur_box is not None:
            line_h = max(box[3] - box[1], 1)
            if box[1] - cur_box[3] > 1.2 * line_h:
                blocks.append(Block(" ".join(cur_text), cur_box, "ocr"))
                cur_text, cur_box = [], None
        cur_text.append(text)
        cur_box = box if cur_box is None else (min(cur_box[0], box[0]), min(cur_box[1], box[1]),
                                               max(cur_box[2], box[2]), max(cur_box[3], box[3]))
    if cur_text:
        blocks.append(Block(" ".join(cur_text), cur_box, "ocr"))
    return blocks


# ----------------------------------------------------------------------------- cleaning
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTISPACE = re.compile(r"[ \t ]+")
_PAGE_NUM = re.compile(r"^\s*(page\s*)?[\divxlcIVXLC]{1,6}(\s*(of|/)\s*\d+)?\s*$", re.I)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                 # soft hyphen
    text = _HYPHEN_BREAK.sub(r"\1\2", text)           # exam-\nple -> example
    text = re.sub(r"\s*\n\s*", " ", text)             # a block is one paragraph
    text = _MULTISPACE.sub(" ", text)
    text = "".join(ch for ch in text if ch.isprintable())
    return text.strip()


def _signature(text: str) -> str:
    """Header/footer fingerprint: digits masked so 'Chapter 3 · 41' == 'Chapter 3 · 42'."""
    return re.sub(r"\d+", "#", text.lower()).strip()


def _in_band(b: Block, p: Page) -> bool:
    """Block starts in the top band or ends in the bottom band of the page."""
    band = config.HEADER_FOOTER_BAND * p.height
    return b.bbox[1] < band or b.bbox[3] > p.height - band


def _strip_headers_footers(pages: list[Page]) -> int:
    """Remove short blocks in the top/bottom band that repeat across pages (running headers/footers,
    digits masked), plus bare page numbers."""
    if len(pages) < 4:
        return 0
    counts: Counter = Counter()
    for p in pages:
        counts.update({_signature(b.text) for b in p.blocks if _in_band(b, p) and len(b.text) < 120})
    min_count = max(3, int(config.HEADER_FOOTER_MIN_FREQ * len(pages)))
    repeated = {s for s, c in counts.items() if c >= min_count}
    removed = 0
    for p in pages:
        keep = []
        for b in p.blocks:
            if _in_band(b, p) and len(b.text) < 120 and (_signature(b.text) in repeated or _PAGE_NUM.match(b.text)):
                removed += 1
                continue
            keep.append(b)
        p.blocks = keep
    return removed


def _detect_lang(text: str) -> str:
    if len(text) < 50:
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text[:2000])
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- main entry
def extract_pdf(path, progress=None) -> tuple[list[Page], dict]:
    """Extract every page. `progress(done, total)` is called after each page."""
    pages: list[Page] = []
    stats = {"pages": 0, "native_pages": 0, "ocr_pages": 0, "ocr_images": 0,
             "headers_removed": 0, "ocr_engine": None}
    with fitz.open(path) as doc:
        total = doc.page_count
        for i, pg in enumerate(doc):
            rect = pg.rect
            page = Page(number=i + 1, width=rect.width, height=rect.height)
            raw = pg.get_text("blocks", sort=True)
            for x0, y0, x1, y1, text, _bno, btype in raw:
                if btype != 0:      # 1 == image block
                    continue
                t = normalize_text(text)
                if t:
                    page.blocks.append(Block(t, (x0, y0, x1, y1)))
            native_chars = sum(len(b.text) for b in page.blocks)

            if config.OCR_ENABLED and native_chars < config.OCR_MIN_PAGE_CHARS:
                # scanned page (or image-only page): OCR the whole page
                ocr_blocks = [Block(normalize_text(b.text), b.bbox, "ocr") for b in _ocr_region(pg)]
                ocr_blocks = [b for b in ocr_blocks if b.text]
                if ocr_blocks:
                    page.blocks = ocr_blocks
                    page.method = "ocr"
                    stats["ocr_pages"] += 1
            elif config.OCR_ENABLED and config.OCR_EMBEDDED_IMAGES:
                # native page: OCR large embedded images (figures, scanned inserts) for their text
                page_area = rect.width * rect.height
                for img in pg.get_image_info():
                    r = fitz.Rect(img["bbox"]) & rect
                    if r.is_empty or r.width * r.height < config.OCR_IMAGE_MIN_AREA * page_area:
                        continue
                    # skip images already covered by native text (e.g. text drawn over a background)
                    covered = sum(len(b.text) for b in page.blocks if fitz.Rect(b.bbox).intersects(r))
                    if covered > 200:
                        continue
                    extra = [Block(normalize_text(b.text), b.bbox, "ocr") for b in _ocr_region(pg, r)]
                    extra = [b for b in extra if len(b.text) > 3]
                    if extra:
                        page.blocks.extend(extra)
                        page.method = "native+ocr"
                        stats["ocr_images"] += 1
            if page.method.startswith("native"):
                stats["native_pages"] += 1
            pages.append(page)
            if progress:
                progress(i + 1, total)

    stats["pages"] = len(pages)
    stats["headers_removed"] = _strip_headers_footers(pages)
    for p in pages:
        p.lang = _detect_lang(p.text)
    langs = Counter(p.lang for p in pages if p.lang != "unknown")
    stats["lang"] = langs.most_common(1)[0][0] if langs else "unknown"
    stats["ocr_engine"] = _ocr.name
    return pages, stats
