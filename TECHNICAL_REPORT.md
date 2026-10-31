# Technical Report: RAG Chatbot over a Large PDF Corpus

**System:** Doc-AI large-corpus RAG engine (`backend/rag`) · **Run date:** 2026-10-31
**Corpus:** 12 PDFs, 4,584 pages, 6,604 chunks · **Hardware:** Intel i5-1135G7, 8 GB RAM, no GPU

Every figure in this report comes from a logged run: `backend/data/logs/ingest.log`, `backend/data/eval/report.json` and `backend/data/logs/queries.jsonl`. The step-by-step reasoning behind each design choice is in [docs/DESIGN_AND_BUILD_GUIDE.md](docs/DESIGN_AND_BUILD_GUIDE.md).

## Summary

| Requirement | Result |
| --- | --- |
| ≥ 10 PDFs, each ≥ 200 pages | 11 real PDFs of 244–640 pages, plus a 30-page image-only scan |
| Native text + OCR | PyMuPDF; RapidOCR on 44 scanned pages and 26 embedded images |
| Chunks of 500–1000 tokens, 10–30% overlap | 500-token max (≈470 avg), 75-token (15%) sentence-aligned overlap |
| Free / OSS embeddings and vector DB | bge-small-en-v1.5 (MIT) · ChromaDB HNSW (Apache-2.0) |
| ANN index | HNSW, cosine, M = 32, ef_construction = 200, ef_search = 100 |
| Reranking | hybrid dense + BM25 with RRF, then a MiniLM cross-encoder |
| Citations | every answer cites PDF and page; 90% of tags valid, 80% cite the gold page |
| Relevance | Recall@3 = 97.5%, Recall@5 = 97.5%, MRR@10 = 0.867 (40 questions) |
| Latency 2–5 s | first word 5.0 s p50 / 6.6 s p95; full answer 6.7 s p50 / 9.7 s p95 (local CPU LLM) |

The 2–5 s target is met for retrieval (0.55 s) and nearly met for time to first word. It is not met for full answers on this CPU-only laptop, because a local 1.5B model generates at about 20 tokens/s. Setting `GROQ_API_KEY` switches generation to a hosted open-weights model (about 1 s) with no code change.

## 1. Embedding model trade-off

| Model | Max input | Dim | Fits a 500-token chunk? | Chosen |
| --- | --- | --- | --- | --- |
| BAAI/bge-small-en-v1.5 | 512 tokens | 384 | yes | **yes** |
| all-MiniLM-L6-v2 | 256 tokens | 384 | no (second half silently dropped) | no |

Both models have the same vector size, so index size and search speed are identical. bge-small was chosen because its 512-token window covers a whole chunk and it is trained for question-to-passage search, with a query instruction.

Measured cost on this CPU: 6,604 chunks were embedded in about 25 minutes (about 4–9 chunks/s depending on load). One query embeds in about 0.1 s.

## 2. Vector DB and index performance

ChromaDB stores 6,604 × 384 vectors (about 10 MB of floats) with an HNSW graph.

| Operation | Measured |
| --- | --- |
| Dense top-30 query (HNSW) | 14–18 ms |
| BM25 top-30 query (sparse matrix) | 5–8 ms |
| Upsert into Chroma during ingestion | 3–47 s per document (42–1,186 chunks) |

At this size HNSW returns near-exact neighbours. IVF+PQ would only pay off at millions of vectors.

## 3. Chunking

Chunks are packed sentence by sentence up to 500 tokens, counted with the embedding model's tokenizer, so no chunk is ever truncated. Each chunk starts with the last ≥ 75 tokens of whole sentences from the previous one. Running headers and footers are removed before chunking: 140 blocks on Think Python alone.

Each chunk keeps its document id (the SHA-1 of the file), filename, pages, a bounding box per page, its source type (native or OCR) and its language. Chunking is deterministic: re-ingesting a file gives identical chunk ids.

## 4. Retrieval quality (40 synthetic questions with gold pages)

| Retriever | R@1 | R@3 | R@5 | R@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| Dense only (HNSW) | 75.0% | 87.5% | 92.5% | 92.5% | 0.821 |
| BM25 only | 85.0% | 97.5% | 97.5% | 100% | 0.907 |
| Hybrid (RRF, k = 60) | 90.0% | 95.0% | 97.5% | 97.5% | 0.930 |
| Hybrid + cross-encoder (deployed) | 80.0% | 97.5% | 97.5% | 97.5% | 0.867 |

Hybrid search fixes dense retrieval's misses: dense alone fails 3 of 40 questions. On this question set the cross-encoder lowers top-1 accuracy. The questions reuse the passage's own words, which favours keyword matching, and the reranker reads only the first 256 tokens of each chunk. Disabling it (`RAG_RERANK_ENABLED=false`) saves about 0.5 s; that change should wait for a human-written question set.

## 5. Latency by component

| Component | p50 | Notes |
| --- | --- | --- |
| Query embedding | ~0.1 s | bge-small on CPU |
| Dense + BM25 + RRF | ~0.03 s | |
| Cross-encoder rerank (12 × 256 tokens) | ~0.45 s | was ~1.1 s at 16 × 384 |
| Retrieval total | 0.55 s | p95 0.79 s |
| First answer word | 5.0 s | p95 6.6 s; dominated by the LLM reading a ~400-token prompt |
| Full answer | 6.7 s | p95 9.7 s |

Tuning history on the same machine (details in the guide, Step 9):

| Configuration | Full answer p50 | First word p50 |
| --- | --- | --- |
| qwen2.5:3b, ~860-token prompt | ~31 s | ~22 s |
| qwen2.5:1.5b, ~470-token prompt, lighter reranker | 8.3 s | 6.8 s |
| plus shorter source labels, 260-token context | 6.2 s (8 questions) / 6.7 s (40-question eval) | 5.0 s |

## 6. Answer quality

| Measure | Result |
| --- | --- |
| Answered (not declined) | 100% |
| Citation accuracy: gold page cited | 80.0% |
| Citation validity: [S#] tags pointing to a real source | 90.0% |
| Hallucination rate (lexical check, > 1/3 unsupported sentences) | 10.0% |
| Average grounding | 90.6% |

Most citation misses point to a neighbouring page or a related standard, for example NIST 800-82 instead of 800-53. The hallucination metric is a lexical proxy. It misses paraphrased errors, so a natural-language-inference (NLI) model or an LLM judge would be stricter.

## 7. Failure cases and mitigations

| Failure | Mitigation in place |
| --- | --- |
| Scanned pages have no text layer | OCR when a page has < 40 native characters; wordninja restores spaces RapidOCR drops |
| Running headers pollute chunks | top/bottom 10% blocks repeated on ≥ 3 pages are removed (digits masked) |
| Exact identifiers missed by vectors | BM25 alongside dense search, fused by RRF |
| Small model omits citation tags | untagged sentences are matched to their best-supporting source; counted separately as `auto_citations` |
| Unsupported claims | per-sentence grounding check; answers flagged when > 1/3 of sentences are unsupported |
| LLM down (Ollama stopped) | extractive answer of the 3 best-matching source sentences, still cited |
| Memory pressure on 8 GB | one local model loaded, fixed `num_ctx`, warm-up at startup |

Open issues: rotated margin text survives cleaning (seen in one book), RapidOCR makes occasional character errors on noisy scans, and the 40-question synthetic eval is a small sample. One question moves a percentage by 2.5 points.
