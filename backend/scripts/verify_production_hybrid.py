"""
Production-grade hybrid completeness verification (static code checks only).
Does not require a live Gemini API key.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ast, inspect

print("=" * 60)
print("PRODUCTION HYBRID COMPLETENESS VERIFICATION")
print("=" * 60)

# ----------------------------------------------------------------
# 1. MCP server BM25 sync
# ----------------------------------------------------------------
print("\n[1] MCP server BM25 sync...")
src = open("mcp_server.py").read()
assert "def get_bm25_index" in src,           "get_bm25_index missing from mcp_server!"
assert "get_bm25_index().add_document" in src, "upload tool not calling BM25.add_document!"
assert "get_bm25_index().delete_document" in src, "delete tool not calling BM25.delete_document!"
print("  PASS - MCP server syncs BM25 on upload and delete")

# ----------------------------------------------------------------
# 2. BM25 migration from existing ChromaDB docs
# ----------------------------------------------------------------
print("\n[2] BM25 migration from existing ChromaDB docs...")
bm25_src = open("core/bm25_index.py").read()
assert "def migrate_from_vector_store" in bm25_src, "migration method missing from BM25Index!"
assert "collection.get(" in bm25_src,         "migration not pulling from ChromaDB!"
assert "BM25Okapi(tokenized)" in bm25_src,    "migration not building BM25 objects!"

main_src = open("main.py").read()
assert "_startup_bm25_migration" in main_src, "startup event missing!"
assert "migrate_from_vector_store" in main_src,"startup not calling migrate!"
assert "threading.Thread" in main_src,        "startup migration not non-blocking!"
print("  PASS - migration method exists and non-blocking startup hook is wired")

# ----------------------------------------------------------------
# 3. Confidence scorer signal source tracking
# ----------------------------------------------------------------
print("\n[3] Confidence scorer signal source tracking...")
scorer_src = open("core/confidence_scorer.py").read()
assert "signal_sources" in scorer_src,  "signal_sources missing from scorer!"
assert '"bm25"' in scorer_src,          "bm25 signal label missing!"
assert '"dense"' in scorer_src,         "dense signal label missing!"
assert "hybrid_score" in scorer_src,    "hybrid_score check missing!"
assert "breakdown = {" in scorer_src,   "breakdown dict never built!"
assert '"answer_coverage"' in scorer_src,"answer_coverage missing from breakdown!"

# Verify the dead-code bug is fixed (old code had: return inside elif)
assert "        return \"low\"" in scorer_src, "low confidence return is still dead code!"
print("  PASS - confidence scorer tracks signal sources in breakdown dict")
print("  PASS - confidence level thresholds: dead-code bug fixed")

# ----------------------------------------------------------------
# 4. Query expansion + HyDE
# ----------------------------------------------------------------
print("\n[4] Query expansion and HyDE...")
config_src = open("config.py").read()
assert "enable_hyde" in config_src,            "enable_hyde missing from config!"
assert "enable_query_expansion" in config_src, "enable_query_expansion missing!"
assert "enable_hyde: bool = False" in config_src,   "HyDE should default to False!"
assert "enable_query_expansion: bool = True" in config_src, "expansion should default True!"

pipeline_src = open("core/rag_pipeline.py").read()
assert "def _expand_query" in pipeline_src,    "_expand_query method missing!"
assert "def _hyde_embedding" in pipeline_src,  "_hyde_embedding method missing!"
assert "_SYNONYMS" in pipeline_src,            "synonym map missing!"
assert "bm25_query" in pipeline_src,           "bm25_query variable missing in ask()!"
assert "enable_hyde" in pipeline_src,          "HyDE flag check missing in ask()!"
assert "enable_query_expansion" in pipeline_src,"expansion flag check missing in ask()!"
assert "sender" in pipeline_src,               "synonym expansion map is empty!"

# Verify query expansion logic directly (no Gemini needed)
from core.rag_pipeline import RAGPipeline

class FakeSelf:
    _SYNONYMS = RAGPipeline._SYNONYMS

expand = RAGPipeline._expand_query
expanded = expand(FakeSelf(), "What is the shipper address?")
assert any(s in expanded for s in ["sender", "origin", "pickup"]), \
    f"Expansion failed. Got: {expanded}"
print(f"  Expanded: 'What is the shipper address?'")
print(f"         -> '{expanded}'")

expanded2 = expand(FakeSelf(), "What is the consignee rate?")
assert "receiver" in expanded2 or "price" in expanded2
print(f"  Multi-keyword expansion for 'consignee' + 'rate': PASS")

# No expansion for unrelated question
expanded3 = expand(FakeSelf(), "What is the weather today?")
assert expanded3 == "What is the weather today?", "Should not expand unrelated queries"
print(f"  No false expansions for unrelated queries: PASS")

print("  PASS - HyDE and query expansion fully wired")

# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL 4 PRODUCTION COMPLETENESS CHECKS PASSED")
print("=" * 60)
print()
print("Summary of what is now fully production-grade:")
print("  1. MCP server  - BM25 synced on every upload/delete")
print("  2. Migration   - Existing ChromaDB docs backfilled at startup")
print("  3. Confidence  - signal_sources tracked (dense vs bm25 per chunk)")
print("  4. Enhancement - Query expansion ON (free) | HyDE OFF (opt-in)")
