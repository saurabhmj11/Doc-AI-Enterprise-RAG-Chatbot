"""
Document Processor — Production RAG Pipeline
=============================================
- Layout-aware PDF extraction with PyMuPDF blocks
- OCR fallback for scanned pages via Tesseract
- Token-based chunking (500-1000 tokens, configurable overlap)
- Sentence-aware overlap boundaries
- Per-chunk metadata: doc_id, filename, page, chunk_index, bbox
"""

import uuid
import re
import logging
import fitz  # PyMuPDF
from docx import Document
import chardet
from typing import Optional
from dataclasses import dataclass, field

from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    page: Optional[int]
    chunk_index: int
    embedding: Optional[list[float]] = None
    bbox: Optional[list[float]] = None  # Bounding box [x0, y0, x1, y1]
    filename: Optional[str] = None


# ============================================================
# TOKEN COUNTER
# ============================================================

_tokenizer = None

def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base). Falls back to word count."""
    global _tokenizer
    try:
        if _tokenizer is None:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(_tokenizer.encode(text))
    except Exception:
        # Fallback: rough word-count approximation
        return len(text.split())


# ============================================================
# OCR HELPER
# ============================================================

def _ocr_page(page: fitz.Page, language: str = "eng") -> str:
    """
    Run OCR on a page using PyMuPDF's built-in Tesseract integration.
    Falls back to pytesseract if PyMuPDF OCR is unavailable.
    """
    try:
        # Try PyMuPDF's built-in OCR first (requires Tesseract installed)
        tp = page.get_textpage_ocr(language=language, dpi=300, full=True)
        text = page.get_text("text", textpage=tp)
        if text.strip():
            return text
    except Exception as e:
        logger.debug(f"PyMuPDF OCR failed, trying pytesseract: {e}")

    try:
        import pytesseract
        from PIL import Image
        import io

        # Render page to image at 300 DPI
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang=language)
        return text
    except ImportError:
        logger.warning("pytesseract not installed, OCR unavailable for this page")
    except Exception as e:
        logger.warning(f"OCR failed for page: {e}")

    return ""


# ============================================================
# PROCESSOR
# ============================================================

class DocumentProcessor:

    def __init__(self):
        self._embedding_model = None
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.use_token_chunking = settings.use_token_chunking

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from core.model_loader import get_embedding_model
            self._embedding_model = get_embedding_model()
        return self._embedding_model


    # ============================================================
    # PDF PARSER WITH OCR
    # ============================================================

    def _parse_pdf(self, file_path: str) -> list[tuple[int, str]]:
        """
        Extract text from PDF with OCR fallback for scanned pages.
        Returns list of (page_number, text) tuples.
        """
        pages = []

        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            logger.info(f"Processing PDF: {total_pages} pages")

            for page_number, page in enumerate(doc, 1):
                # Try native text extraction first (block-level)
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))  # Top-to-bottom, left-to-right

                page_text_parts = []
                for block in blocks:
                    text = block[4].strip()
                    if text:
                        page_text_parts.append(text)

                page_text = "\n".join(page_text_parts)

                # OCR fallback: if native text is too sparse, try OCR
                if settings.enable_ocr and len(page_text.strip()) < 50:
                    logger.debug(f"Page {page_number}: sparse text ({len(page_text)} chars), running OCR")
                    ocr_text = _ocr_page(page, settings.ocr_language)
                    if ocr_text.strip():
                        page_text = ocr_text

                # Clean common artifacts
                page_text = self._clean_text(page_text)

                if page_text.strip():
                    pages.append((page_number, page_text))

                # Log progress for large documents
                if page_number % 50 == 0:
                    logger.info(f"  Processed {page_number}/{total_pages} pages")

        logger.info(f"PDF extraction complete: {len(pages)} pages with content")
        return pages

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove common header/footer patterns
        text = re.sub(r'(?m)^Page \d+ of \d+$', '', text)
        text = re.sub(r'(?m)^[-_]{10,}$', '', text)
        return text.strip()


    # ============================================================
    # TOKEN-BASED CHUNKING
    # ============================================================

    def _create_chunks(self, doc_id: str, pages: list[tuple[int, str]], filename: str = None) -> list[DocumentChunk]:
        """
        Create chunks from pages using token-based splitting.
        Target: 500-1000 tokens per chunk with 10-30% overlap.
        """
        chunks = []
        idx = 0

        for page_num, page_text in pages:
            # Split page into paragraphs
            paragraphs = re.split(r'\n{2,}', page_text)

            current_text = ""
            current_tokens = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                para_tokens = _count_tokens(para) if self.use_token_chunking else len(para)

                # If adding this paragraph exceeds chunk size, finalize current chunk
                if current_tokens + para_tokens > self.chunk_size and current_text:
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}_{idx}",
                        document_id=doc_id,
                        text=current_text.strip(),
                        page=page_num,
                        chunk_index=idx,
                        filename=filename,
                    ))
                    idx += 1

                    # Sentence-aware overlap: take last N tokens worth of text
                    sentences = re.split(r'(?<=[.!?])\s+', current_text)
                    overlap_text = ""
                    overlap_tokens = 0
                    for sent in reversed(sentences):
                        sent_tokens = _count_tokens(sent) if self.use_token_chunking else len(sent)
                        if overlap_tokens + sent_tokens > self.chunk_overlap:
                            break
                        overlap_text = sent + " " + overlap_text
                        overlap_tokens += sent_tokens

                    current_text = overlap_text.strip() + "\n" + para
                    current_tokens = overlap_tokens + para_tokens
                else:
                    current_text = (current_text + "\n" + para) if current_text else para
                    current_tokens += para_tokens

            # Flush remaining text for this page
            if current_text.strip():
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_{idx}",
                    document_id=doc_id,
                    text=current_text.strip(),
                    page=page_num,
                    chunk_index=idx,
                    filename=filename,
                ))
                idx += 1

        logger.info(f"Created {len(chunks)} chunks (avg ~{sum(_count_tokens(c.text) for c in chunks) // max(len(chunks), 1)} tokens/chunk)")
        return chunks


    # ============================================================
    # EMBEDDINGS
    # ============================================================

    def _generate_embeddings(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Generate embeddings for all chunks using the configured model."""
        texts = [c.text for c in chunks]

        # Batch embedding for efficiency
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, convert_to_numpy=True)
            if embeddings is None:
                raise RuntimeError("Embedding generation failed")
            all_embeddings.extend(embeddings)

            if (i + batch_size) % 256 == 0:
                logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

        for chunk, emb in zip(chunks, all_embeddings):
            chunk.embedding = emb.tolist()

        return chunks


    # ============================================================
    # DOCX / TXT PARSERS
    # ============================================================

    def _parse_docx(self, file_path: str) -> list[tuple[int, str]]:
        doc = Document(file_path)
        parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)

        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        full_text = "\n".join(parts)
        return [(1, full_text)] if full_text else []

    def _parse_txt(self, file_path: str) -> list[tuple[int, str]]:
        with open(file_path, "rb") as f:
            raw = f.read()

        enc = chardet.detect(raw).get("encoding", "utf-8") or "utf-8"
        text = raw.decode(enc, errors="replace")
        return [(1, text)] if text.strip() else []

    def get_full_text(self, file_path: str, file_type: str) -> str:
        """Get the full text of a document (used by structured extractor)."""
        if file_type == "pdf":
            pages = self._parse_pdf(file_path)
        elif file_type == "docx":
            pages = self._parse_docx(file_path)
        elif file_type == "txt":
            pages = self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return "\n\n".join(text for _, text in pages)


    # ============================================================
    # MAIN ENTRY
    # ============================================================

    def process_file(self, file_path: str, file_type: str, filename: str = None) -> tuple[str, list[DocumentChunk]]:
        """
        Process a document file into embedded chunks.
        
        Returns:
            (document_id, list of DocumentChunk with embeddings)
        """
        import time
        start = time.time()

        if file_type == "pdf":
            pages = self._parse_pdf(file_path)
        elif file_type == "docx":
            pages = self._parse_docx(file_path)
        elif file_type == "txt":
            pages = self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        doc_id = str(uuid.uuid4())
        chunks = self._create_chunks(doc_id, pages, filename=filename)
        chunks = self._generate_embeddings(chunks)

        elapsed = time.time() - start
        logger.info(
            f"Document processed: {len(pages)} pages → {len(chunks)} chunks "
            f"in {elapsed:.1f}s ({elapsed/max(len(pages),1):.2f}s/page)"
        )

        return doc_id, chunks
