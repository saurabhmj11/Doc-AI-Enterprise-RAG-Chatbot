"""Download a corpus of freely-licensed / public-domain PDFs (each >= 200 pages).

Usage (from backend/):  python scripts/download_corpus.py [--min-pages 200] [--target 12]

Every candidate is verified after download (valid PDF, page count >= min pages);
failures are skipped so the script is robust to individual dead links.
"""
import argparse
import sys
from pathlib import Path

import pymupdf as fitz
import httpx

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "pdfs"

# (filename, url) -- public-domain US government documents and free-to-download textbooks.
CANDIDATES = [
    ("NIST_SP_800-53r5_Security_Controls.pdf",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf"),
    ("NIST_SP_800-82r3_OT_Security.pdf",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf"),
    ("NIST_SP_800-160v1r1_Systems_Security_Engineering.pdf",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-160v1r1.pdf"),
    ("NASA_Systems_Engineering_Handbook.pdf",
     "https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf"),
    ("Think_Python_2e.pdf",
     "https://greenteapress.com/thinkpython2/thinkpython2.pdf"),
    ("Mathematics_for_Machine_Learning.pdf",
     "https://mml-book.github.io/book/mml-book.pdf"),
    ("Open_Data_Structures_Python.pdf",
     "https://opendatastructures.org/ods-python.pdf"),
    ("Foundations_of_Data_Science.pdf",
     "https://www.cs.cornell.edu/jeh/book.pdf"),
    ("Algorithms_Erickson.pdf",
     "https://jeffe.cs.illinois.edu/teaching/algorithms/book/Algorithms-JeffE.pdf"),
    ("Reinforcement_Learning_Sutton_Barto.pdf",
     "http://incompleteideas.net/book/RLbook2020.pdf"),
    ("Information_Theory_MacKay.pdf",
     "https://www.inference.org.uk/itprnn/book.pdf"),
    ("Think_Stats_2e.pdf",
     "https://greenteapress.com/thinkstats2/thinkstats2.pdf"),
    ("NIST_SP_800-37r2_Risk_Management_Framework.pdf",
     "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-37r2.pdf"),
    ("Bayesian_Reasoning_and_ML_Barber.pdf",
     "http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf"),
]


def page_count(path: Path) -> int:
    try:
        with fitz.open(path) as doc:
            return doc.page_count
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pages", type=int, default=200)
    ap.add_argument("--target", type=int, default=12)
    args = ap.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ok = []
    headers = {"User-Agent": "Mozilla/5.0 (rag-chatbot corpus downloader)"}
    with httpx.Client(follow_redirects=True, timeout=120, headers=headers) as client:
        for name, url in CANDIDATES:
            if len(ok) >= args.target:
                break
            dest = PDF_DIR / name
            if not dest.exists():
                try:
                    with client.stream("GET", url) as r:
                        r.raise_for_status()
                        tmp = dest.with_suffix(".part")
                        with open(tmp, "wb") as f:
                            for block in r.iter_bytes(1 << 16):
                                f.write(block)
                    tmp.replace(dest)
                except Exception as e:
                    print(f"[skip] {name}: download failed ({e})")
                    continue
            n = page_count(dest)
            if n < args.min_pages:
                print(f"[skip] {name}: {n} pages (< {args.min_pages})")
                dest.unlink(missing_ok=True)
                continue
            size_mb = dest.stat().st_size / 1e6
            print(f"[ok]   {name}: {n} pages, {size_mb:.1f} MB")
            ok.append(name)

    print(f"\n{len(ok)} PDFs ready in {PDF_DIR}")
    if len(ok) < 10:
        sys.exit("fewer than 10 PDFs downloaded - add candidates")


if __name__ == "__main__":
    main()
