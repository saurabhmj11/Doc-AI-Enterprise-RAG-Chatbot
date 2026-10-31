"""Create an image-only ("scanned") PDF to exercise the OCR path.

Downloads NIST SP 800-37r2 (public domain), rasterises the first N pages to grayscale images with a
slight rotation + noise (like a photocopy), and writes them as an image-only PDF with no text layer.

Usage: python scripts/make_scanned_sample.py [--pages 30]
"""
import argparse
import io
import random
from pathlib import Path

import httpx
import pymupdf
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
URL = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-37r2.pdf"
OUT = ROOT / "data" / "pdfs" / "SCANNED_NIST_SP_800-37r2_RMF_excerpt.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--start", type=int, default=8)
    args = ap.parse_args()
    random.seed(0)

    src_bytes = httpx.get(URL, follow_redirects=True, timeout=120).content
    src = pymupdf.open(stream=src_bytes, filetype="pdf")
    out = pymupdf.open()
    for i in range(args.start, min(args.start + args.pages, src.page_count)):
        page = src[i]
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        img = img.rotate(random.uniform(-0.6, 0.6), expand=False, fillcolor=255)
        img = img.filter(ImageFilter.GaussianBlur(0.4))
        noise = Image.effect_noise(img.size, 12).convert("L")
        img = Image.blend(img, noise, 0.06)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    out.save(OUT, garbage=4, deflate=True)
    print(f"wrote {OUT} ({out.page_count} image-only pages)")


if __name__ == "__main__":
    main()
