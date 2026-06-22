"""Smoke test for BM25 + Hybrid Fuser integration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.bm25_index import BM25Index
from core.hybrid_fuser import HybridFuser
from core.document_processor import DocumentChunk

# ---- Build a tiny fake document ----
doc_id = "smoke_test_001"
chunks = [
    DocumentChunk(
        chunk_id=f"{doc_id}_0", document_id=doc_id,
        text="Shipper: AAA Logistics, Los Angeles, CA. Rate: 2850 USD.",
        page=1, chunk_index=0, embedding=[0.1] * 384,
    ),
    DocumentChunk(
        chunk_id=f"{doc_id}_1", document_id=doc_id,
        text="Consignee: XYZ Distribution, Chicago, IL. Delivery 02-12-2026.",
        page=1, chunk_index=1, embedding=[0.2] * 384,
    ),
    DocumentChunk(
        chunk_id=f"{doc_id}_2", document_id=doc_id,
        text="Load ID LD53657, equipment type: Dry Van, weight 24000 lbs.",
        page=2, chunk_index=2, embedding=[0.3] * 384,
    ),
]

# ---- BM25 index ----
print("=== BM25 Search ===")
bm25 = BM25Index()
bm25.add_document(doc_id, chunks)

sparse_results = bm25.search(doc_id, "What is the Load ID?", top_k=3)
for r in sparse_results:
    print(f"  {r['chunk_id']} | bm25_score={r['bm25_score']:.4f} | text={r['text'][:60]}")

assert "LD53657" in sparse_results[0]["text"], "BM25 did not rank Load ID chunk first!"
print("BM25 correctly ranked 'Load ID LD53657' chunk at position 1")

# ---- Simulate dense results (different ranking) ----
dense_results = [
    {"chunk_id": f"{doc_id}_0", "text": chunks[0].text, "similarity_score": 0.9, "page": 1, "filename": "test.pdf"},
    {"chunk_id": f"{doc_id}_1", "text": chunks[1].text, "similarity_score": 0.7, "page": 1, "filename": "test.pdf"},
    {"chunk_id": f"{doc_id}_2", "text": chunks[2].text, "similarity_score": 0.5, "page": 2, "filename": "test.pdf"},
]

# ---- RRF Fusion ----
print("\n=== RRF Fusion ===")
fuser_rrf = HybridFuser(strategy="rrf")
fused_rrf = fuser_rrf.fuse(dense_results, sparse_results, top_k=3)
for r in fused_rrf:
    print(f"  {r['chunk_id']} | hybrid_score={r['hybrid_score']}")

assert len(fused_rrf) == 3, "RRF fusion should return 3 chunks"

# ---- Alpha-Blend Fusion ----
print("\n=== Alpha-Blend Fusion ===")
fuser_alpha = HybridFuser(strategy="alpha", alpha=0.5)
fused_alpha = fuser_alpha.fuse(dense_results, sparse_results, top_k=3)
for r in fused_alpha:
    print(f"  {r['chunk_id']} | hybrid_score={r['hybrid_score']}")

assert len(fused_alpha) == 3, "Alpha fusion should return 3 chunks"

print("\nSMOKE TEST PASSED ✅")
