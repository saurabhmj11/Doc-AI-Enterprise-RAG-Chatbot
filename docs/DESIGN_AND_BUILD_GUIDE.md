# Doc-AI Large-Corpus RAG Chatbot — Design & Build Guide

Oct 31, 2026 · Saurabh Lokhande

> Living version: https://claude.ai/artifact/GxhV7XhC9XJzSan61UWfm9 (this file is an export for the repository).

## Overview

Doc-AI now answers questions over 12 real PDFs (4,584 pages) using only free, open-source models and a local LLM, with a citation to file and page on every answer. On a CPU-only laptop the first answer word arrives in about 5.0 s (median) and full answers finish in about 6.7 s. The right page is in the top 3 results for 97.5% of test questions.

**The challenge.** Build a Retrieval-Augmented Generation (RAG) chatbot over at least 10 PDFs of 200+ pages each. It must OCR scanned pages, use a free embedding model and vector DB, answer in 2–5 s, cite PDF name and page, and report latency, recall, MRR, hallucination and citation accuracy.

**What RAG means here.** An LLM alone answers from memory and invents facts. RAG first *retrieves* the few passages of your documents that match the question, then asks the LLM to answer *only from those passages*. Every claim can then be traced to a page.

**What was built, in one line per part:**

- An ingestion pipeline: PDF → text and OCR → cleaning → chunks → embeddings → vector DB.
- A hybrid retriever: vector search plus keyword search, fused, then reranked by a second model.
- A cited, streaming answer generator with an automatic grounding check and an offline fallback.
- A React UI inside Doc-AI showing the chat, the retrieved chunks, the ingestion pipeline and the metrics.
- An evaluation harness that measures recall, MRR, latency, citation accuracy and hallucination rate.

| Measure | Result | Target |
| --- | --- | --- |
| Corpus | 12 PDFs, 4,584 pages, 6,604 chunks | ≥ 10 PDFs × 200 pages |
| Retrieval time (p50 / p95) | 0.55 s / 0.79 s | — |
| First word of the answer (p50 / p95) | 5.0 s / 6.6 s | 2–5 s |
| Full answer (p50 / p95) | 6.7 s / 9.7 s | 2–5 s |
| Cost | $0, everything runs locally | free / open source |

The full answer still runs about 1–2 s over the target because the LLM runs on a laptop CPU with no GPU. Step 9 explains the measurements and the one-line switch to a hosted model that brings generation under 2 s.

## Architecture at a glance

```
Ingestion · once per PDF
  PDFs ──► Extract + OCR ──► Clean ──► Chunk ──────────────► Embed
  12 files  PyMuPDF text      headers,   500 tokens            bge-small
  4,584 pp  RapidOCR scans    hyphens,   75 overlap            384 numbers
                              language        │                   │
                                              ▼                   ▼
Shared index on disk              ┌────────────────────┐ ┌──────────────────────────┐
                                  │ BM25 keyword index │ │ ChromaDB · HNSW vectors  │
                                  │ 6,604 chunks       │ │ 6,604 × 384, cosine      │
                                  └─────────┬──────────┘ └────────────┬─────────────┘
                                            └──────────┬──────────────┘
Question answering · per query                         ▼
  Question ──► Hybrid search ──► RRF fusion ──► Rerank ─────────► LLM answer
  embedded     top 30 + top 30   merge by rank  cross-encoder     qwen2.5:1.5b
  by bge       ~20 ms            k = 60         best 12 → top 3   cited + checked
```

Ingestion runs once per PDF and writes two indexes of the same 6,604 chunks; every question searches both, merges the results, reranks them and hands the best three passages to the LLM.

## Step 1 — The corpus: 12 real PDFs instead of generated filler

The system is tested on 11 real, freely downloadable books and government standards (244–640 pages each) plus one deliberately scanned PDF. `backend/scripts/download_corpus.py` fetches them and rejects any file under 200 pages.

**Why real documents.** The earlier Doc-AI corpus was generated from templates: page 50 and page 150 of a file differed only in a section number. Retrieval over repeated text looks almost perfect and proves nothing. Real books have tables, equations, code, figures, running headers and overlapping topics, which is what a private corpus looks like.

**Why these titles.** They span six unrelated domains (security standards, systems engineering, programming, statistics, algorithms, machine learning). Questions in one domain must not retrieve pages from another, so the mix tests precision. All are public domain or free to download, so anyone can reproduce the results.

| File | Pages | Domain |
| --- | --- | --- |
| Information_Theory_MacKay.pdf | 640 | information theory |
| Reinforcement_Learning_Sutton_Barto.pdf | 548 | machine learning |
| NIST_SP_800-53r5_Security_Controls.pdf | 492 | security standard |
| Foundations_of_Data_Science.pdf | 479 | data science |
| Algorithms_Erickson.pdf | 472 | algorithms |
| Mathematics_for_Machine_Learning.pdf | 417 | mathematics |
| NASA_Systems_Engineering_Handbook.pdf | 356 | systems engineering |
| Open_Data_Structures_Python.pdf | 326 | data structures |
| NIST_SP_800-82r3_OT_Security.pdf | 316 | security standard |
| Think_Stats_2e.pdf | 264 | statistics |
| Think_Python_2e.pdf | 244 | programming |
| SCANNED_NIST_SP_800-37r2_RMF_excerpt.pdf | 30 | security (image-only scan) |

**Why a scanned file.** The challenge requires OCR, but none of the downloaded books are scans. `backend/scripts/make_scanned_sample.py` renders 30 pages of NIST SP 800-37 to grayscale images, adds slight rotation, blur and noise like a photocopy, and saves them with no text layer. The only way to read that file is OCR, so it proves the OCR path works end to end.

## Step 2 — Extraction and OCR: native text first, OCR only where needed

Every page is read with PyMuPDF; pages with almost no text are OCR'd, and so are large images on normal pages. Across the corpus this OCR'd 44 whole pages and 26 embedded images. Code: `backend/rag/extract.py`.

**1. Native text with PyMuPDF.** `page.get_text("blocks", sort=True)` returns paragraphs with their bounding box (x0, y0, x1, y1) in reading order. *Why PyMuPDF:* it is the fastest open-source PDF parser (about 50 pages per second here), and it gives coordinates, which later let the UI highlight the exact cited paragraph on the page. pdfminer and pypdf are slower and give weaker layout.

**2. Detecting scanned pages.** A page with fewer than 40 native characters (`OCR_MIN_PAGE_CHARS`) is treated as scanned. *Why a character threshold:* a scan has an image but no text layer, so this test is cheap, reliable and needs no model.

**3. OCR engine: RapidOCR, or Tesseract when installed.** The page is rendered to a 200-DPI grayscale image and passed to the OCR engine. Detected lines are converted back to PDF coordinates and merged into paragraphs. *Why RapidOCR by default:* Tesseract was not installed on this machine and needs a separate system installer; RapidOCR (PaddleOCR models in ONNX format, Apache-2.0) installs with `pip` alone and runs on CPU. If `tesseract` is on the PATH, the code uses it automatically.

**4. Restoring spaces with wordninja.** RapidOCR's multilingual model often drops spaces in English ("Tofacilitatetheintegration…"). Any run of 14+ letters is re-split with `wordninja`, a small word-frequency model. *Why:* without spaces, keyword search and embeddings both fail on OCR'd text. After the fix the scanned NIST pages read as normal sentences.

**5. OCR of embedded images.** On normal pages, an image covering at least 20% of the page that has little text over it is OCR'd and its text added as extra blocks. *Why:* diagrams and inserted scans often hold labels or tables that are not in the text layer. The 20% floor skips logos and icons, which keeps ingestion fast.

## Step 3 — Cleaning: remove what is not content

Extracted text is normalised, running headers and footers are deleted, and each page's language is detected. On Think Python alone this removed 140 header and footer blocks.

- **Unicode normalisation (NFKC).** Turns ligatures such as "ﬁ" into "fi" and odd spaces into normal ones. *Why:* otherwise "ﬁle" and "file" are different words to the keyword index.
- **De-hyphenation.** "exam-⏎ple" becomes "example", and soft hyphens are dropped. *Why:* words split across lines would never match a query.
- **Header and footer removal.** Blocks in the top or bottom 10% of a page are fingerprinted with digits masked, so "Chapter 5 · 40" and "Chapter 5 · 42" match. A fingerprint seen on 3+ pages (or 2% of pages) is deleted, along with bare page numbers. *Why:* a running header repeats on hundreds of pages. Left in, it pollutes every chunk and makes unrelated pages look similar to the search.
- **Language detection (langdetect, fixed seed).** Each page gets a language code stored with its chunks. *Why:* the challenge asks for it, and it lets a future version route non-English pages to a multilingual model. All 12 documents detected as English.

*Known gap:* MacKay's book prints a copyright notice sideways in the margin ("On-screen viewing permitted…"). It sits outside the top and bottom bands, so it survives cleaning and appears in some chunks. The limitations section lists the fix.

## Step 4 — Chunking: 500-token passages with 75-token overlap

Text is split into passages of at most 500 tokens, each starting with the last ~75 tokens of the previous one (15% overlap). The 4,584 pages became 6,604 chunks averaging about 470 tokens. Code: `backend/rag/chunking.py`.

**How it works.**

1. Each page's paragraphs are split into sentences.
2. Sentences are counted in tokens using the embedding model's own tokenizer.
3. Sentences are packed into a chunk until the next one would pass 500 tokens.
4. The next chunk restarts with the trailing whole sentences of the previous one, totalling at least 75 tokens.
5. A single sentence longer than 500 tokens (a long table row, say) is split on word boundaries.

**Why these numbers.**

- *500 tokens:* the challenge asks for 500–1000. The embedding model reads at most 512 tokens and silently cuts the rest. 500 is the largest size that is never truncated. The old Doc-AI setting (800 tokens with a 256-token model) threw away most of every chunk.
- *15% overlap, sentence-aligned:* an answer that straddles a chunk boundary still appears whole in one chunk. Overlapping on whole sentences avoids half-sentences that confuse both search and the LLM.
- *Counted with the real tokenizer:* word counts misjudge technical text with code and symbols; the tokenizer is exact.

**Metadata kept on every chunk.** Document id, filename, chunk index, first and last page, the full page list, a bounding box per page, source type (native, ocr or mixed), language and token count. *Why:* the citation shows filename and pages, and the bounding box lets the UI draw a highlight on the page image.

**Determinism.** The document id is the SHA-1 hash of the file's bytes, and a chunk id is `<doc id>:<index>`. Re-running ingestion on the same file gives identical chunks and ids, and unchanged files are skipped. *Why:* the challenge asks for reproducibility, and stable ids keep the evaluation's gold answers valid.

## Step 5 — Embeddings: BAAI/bge-small-en-v1.5

Every chunk is turned into a 384-number vector by `BAAI/bge-small-en-v1.5`, an MIT-licensed model that runs on the CPU. Embedding all 6,604 chunks took about 25 minutes; embedding one question takes about 0.1 s.

**What an embedding is.** A vector that places text by meaning, so "How are TD and Monte Carlo methods different?" lands near a passage comparing them, even with different wording.

**Why bge-small rather than all-MiniLM-L6-v2 (the old Doc-AI model).**

|  | bge-small-en-v1.5 | all-MiniLM-L6-v2 |
| --- | --- | --- |
| Max input | 512 tokens | 256 tokens |
| Vector size | 384 | 384 |
| Parameters | 33 M | 22 M |
| Built for search? | Yes: separate query instruction | General sentence similarity |
| Licence | MIT | Apache-2.0 |

The deciding factor is input length: MiniLM would read only the first half of each 500-token chunk. Both produce 384-number vectors, so storage and search speed are the same.

**Query instruction.** Questions are prefixed with "Represent this sentence for searching relevant passages:" before embedding; passages are not. *Why:* bge was trained this way, and it improves question-to-passage matching.

**Normalised vectors, saved to disk.** Vectors are scaled to length 1, so cosine similarity is a plain dot product. Each document's vectors are also saved as `data/index/embeddings/<doc id>.npy`. *Why:* the challenge asks for precomputed embeddings, and the index can be rebuilt without re-running the model.

## Step 6 — Indexing: ChromaDB (HNSW) for meaning, BM25 for exact words

The 6,604 vectors live in ChromaDB with an HNSW index, and the same chunks also go into a BM25 keyword index. A vector lookup takes about 15 ms and a keyword lookup about 5 ms. Code: `backend/rag/store.py`.

**ChromaDB.** An open-source (Apache-2.0) vector database that stores vectors, text and metadata in a local folder. *Why:* free, no server to run, persists to disk, and Doc-AI already used it. Embeddings are supplied by our code rather than Chroma's built-in model, so the model choice stays explicit.

**HNSW index settings.** HNSW (Hierarchical Navigable Small World) is a graph that finds the nearest vectors without comparing against all of them.

| Setting | Value | Effect |
| --- | --- | --- |
| space | cosine | matches the normalised bge vectors |
| M (links per node) | 32 | denser graph, better recall, a little more memory |
| ef_construction | 200 | higher-quality graph built at ingest time |
| ef_search | 100 | candidates explored per query: near-exact recall at this size |

*Why HNSW and not IVF+PQ:* at 6,604 vectors HNSW is already fast and close to exact. IVF+PQ compresses vectors to save memory, which only pays off at millions of vectors and costs some accuracy.

**BM25 keyword index.** BM25 scores a chunk by how often the query's rare words appear in it, adjusted for chunk length. It is implemented in about 60 lines as a sparse matrix (SciPy), so one query is a single matrix multiply. *Why add keywords at all:* vectors are weak on exact identifiers such as "IR-4", "SP 800-53" or a function name. BM25 catches those, and the two methods fail on different questions.

*Why our own BM25 instead of the rank-bm25 package:* rank-bm25 loops in Python over every chunk for every query word; the sparse-matrix version does the same maths in one vectorised step.

## Step 7 — Retrieval: hybrid search, rank fusion, then reranking

A question runs through both indexes, the two ranked lists are merged, and a second model re-scores the best 12 before the top few go to the LLM. The whole step takes about 0.55 s (median), most of it reranking. Code: `backend/rag/retrieve.py`.

1. **Embed the question** (about 0.1 s) with the bge query instruction.
2. **Dense search:** top 30 chunks from the HNSW index by cosine similarity.
3. **Keyword search:** top 30 chunks from BM25.
4. **Reciprocal Rank Fusion (RRF, k = 60).** Each chunk scores the sum of 1 / (60 + its rank) over the lists it appears in. *Why RRF:* cosine scores (0–1) and BM25 scores (0–30+) are on different scales and cannot be added. RRF uses only ranks, needs no tuning, and rewards chunks found by both methods.
5. **Cross-encoder reranking** of the 12 best fused chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0). It reads question and chunk *together* and outputs a relevance score. *Why:* the embedding model compares two separately made vectors, which is fast but approximate. A cross-encoder sees both texts at once and is much more precise, but too slow to run on all 6,604 chunks. So it only re-orders a short list.
6. **Select the context:** the top 3 reranked chunks go to the local LLM (top 5 for a hosted LLM).

**Speed settings and why.** The reranker reads 256 tokens per pair (`RERANK_MAX_LEN`), down from 384, and scores 12 candidates, down from 16. Together they cut reranking from about 1.1 s to about 0.5 s; the start of a chunk carries most of its relevance signal.

**Everything is recorded for the UI.** For each candidate the trace keeps its dense rank and score, BM25 rank and score, RRF rank, rerank score, and whether it went into the prompt. The chat screen's "Retrieval trace" panel draws these as bars and a latency waterfall.

## Step 8 — Generation: a short, cited answer that is checked afterwards

The local model `qwen2.5:1.5b` (via Ollama, Apache-2.0) writes a short answer from the selected passages and tags each sentence with its source, like [S1]. The tags are then verified and mapped to filename and page. Code: `backend/rag/pipeline.py` and `backend/rag/llm.py`.

**1. Context building.** Each selected chunk is trimmed to the sentences that share the most words with the question, keeping their original order, until about 260 tokens in total. Each is labelled [S1], [S2], [S3]. *Why:* on a CPU every prompt token costs about 7 ms before the first answer word, so the prompt carries only the sentences most likely to hold the answer.

**2. The prompt.** A three-sentence instruction: answer only from the numbered sources, at most 45 words, end every sentence with its source tag, and if the sources do not contain the answer reply exactly "I could not find this in the indexed documents." *Why so short:* the earlier long prompt cost seconds. Ending the request with "Answer with [S#] citations:" also made small models reply with just "[S1]", so it now ends with "Answer:".

**3. Streaming.** Tokens are sent to the browser as they are generated (server-sent events). *Why:* the reader sees the answer start after about 5 s instead of waiting for the whole of it.

**4. Citation repair.** Small models sometimes forget a tag. Each untagged sentence is matched to the source sharing at least half of its content words, and that tag is added. These automatic tags are counted separately (`auto_citations`) so the evaluation does not credit them to the model.

**5. Verification.** For every sentence the code measures how many of its content words appear in the source it cites. Below 50% the sentence is marked unsupported. More than a third unsupported raises a hallucination flag shown under the answer. *Why:* a cheap, instant check on every live answer; Step 10 notes its limits.

**6. Mapping to pages.** [S1] becomes, for example, "Reinforcement_Learning_Sutton_Barto.pdf · p. 146". Clicking it opens the rendered page with the chunk's bounding box highlighted.

**7. Offline fallback.** If Ollama is stopped or unreachable, the answer is built from the three source sentences that best match the question, each still tagged. *Why:* the chat keeps working, and stays cited, even with no LLM.

## Step 9 — Latency tuning on a CPU-only laptop

Median answer time fell from about 31 s to 6.2 s in an 8-question benchmark (6.7 s on the 40-question eval), mostly from a smaller model and a shorter prompt. The machine is an Intel i5-1135G7 (4 cores) with 8 GB of RAM and no GPU.

**Where the time went at the start.** Retrieval took about 1.2 s. The LLM (`qwen2.5:3b`) spent 22–30 s just *reading* the prompt before writing a word. Prompt reading speed, not writing, was the bottleneck.

**Model comparison, measured on this laptop with real RAG prompts:**

| Model | Reads prompt (tokens/s) | Writes (tokens/s) | Answer quality |
| --- | --- | --- | --- |
| qwen2.5:0.5b | 135–280 | 25–31 | poor: copies text verbatim |
| qwen2.5:1.5b (chosen) | 125–180 | 17–25 | correct, concise, some missed tags |
| qwen2.5:3b | 33–90 | 5–13 | best, but ≥ 9 s even with a tiny prompt |

**Changes made, in the order they were tried:**

| Change | Effect | Answer p50 | First word p50 |
| --- | --- | --- | --- |
| Start: qwen2.5:3b, ~860-token prompt, 16 reranked at 384 tokens | — | ~31 s | ~22 s |
| 1.5B model, 300-token context, 45-word answers, short instruction, reranker 12 × 256 | prompt ~470 tokens | 8.3 s | 6.8 s |
| Unloaded the idle 3B model (2.2 GB) and restarted the backend | stopped memory swapping | (included above) | (included above) |
| Dropped filenames from source labels, 260-token context | prompt ~400 tokens | 6.2 s | 5.0 s |

**Things measured and rejected.** Pinning Ollama to 3 or 4 CPU threads made no measurable difference, so it is left on automatic. Ollama's prompt cache reuses the unchanged instruction prefix; that saves a little but cannot help the passages, which differ per question.

**Memory matters on 8 GB.** At one point only 0.85 GB was free and Windows was compressing memory. The backend had been swapped out, adding 2–3 s per request. A fixed `num_ctx` of 2048 also avoids Ollama reloading the model between calls.

**Reaching under 5 s for the full answer** needs faster generation hardware. Setting `GROQ_API_KEY` in `backend/.env` switches to hosted Llama 3.1 8B (open weights) with a 5-chunk, 2,500-token context; generation then takes about 1 s. No code change is needed.

## Step 10 — Evaluation and monitoring

On 40 test questions the right page was in the top 3 results 97.5% of the time (MRR 0.867). 80% of answers cited the correct page and 10% were flagged as possible hallucinations. The first word arrived at 5.0 s (median) and full answers took 6.7 s (median). Run on 2026-10-31 with the local qwen2.5:1.5b.

**How the test set was made.** `scripts/build_eval.py` picks 40 chunks with a fixed seed, spread evenly over all 12 documents, skipping front and back matter and table-heavy text. For each chunk, `qwen2.5:3b` writes one specific question answerable from it. The chunk's document and pages are the gold answer. *Why a different model writes the questions:* so the answering model is not grading its own phrasing.

**Metrics, and how each is computed (`scripts/run_eval.py`):**

- **Recall@k:** the share of questions where a chunk from the gold document, overlapping the gold pages, is among the top k results.
- **MRR@10:** the average of 1 / rank of the first correct chunk (0 if not in the top 10). 1.0 means always first.
- **Citation accuracy:** the share of answers that cite the gold page.
- **Citation validity:** the share of [S#] tags that point to a real source.
- **Hallucination rate:** the share of answers where more than a third of factual sentences share under half their content words with the cited source.
- **Latency:** p50 and p95 of retrieval, first word and full answer, timed inside the pipeline.

**Retriever comparison (same 40 questions):**

| Retriever | R@1 | R@3 | R@5 | R@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| Dense only (HNSW) | 75.0% | 87.5% | 92.5% | 92.5% | 0.821 |
| BM25 only | 85.0% | 97.5% | 97.5% | 100% | 0.907 |
| Hybrid (RRF) | 90.0% | 95.0% | 97.5% | 97.5% | 0.930 |
| Hybrid + cross-encoder (in use) | 80.0% | 97.5% | 97.5% | 97.5% | 0.867 |

**Answer quality and latency:**

| Measure | Result |
| --- | --- |
| Answered (not declined) | 100% |
| Citation accuracy (gold page cited) | 80.0% |
| Citation validity | 90.0% |
| Hallucination rate | 10.0% |
| Average grounding | 90.6% |
| Retrieval p50 / p95 | 0.55 s / 0.79 s |
| First word p50 / p95 | 5.0 s / 6.6 s |
| Full answer p50 / p95 | 6.7 s / 9.7 s |

**What the numbers say.**

- Hybrid search beats either method alone: dense search misses 3 of 40 questions entirely.
- On this test set the cross-encoder *lowers* top-1 accuracy (90% → 80%), while keeping top-3 at 97.5%. The questions were written from each passage's own words, which favours keyword matching. The reranker also reads only the first 256 tokens of a chunk. Turning it off (`RAG_RERANK_ENABLED=false`) would save about 0.5 s; this should be confirmed on human-written questions first.
- Most citation misses cite a neighbouring page or a related document (for example NIST 800-82 instead of 800-53), not invented content.
- No answer met 5 s end to end on this CPU; Step 9 explains why.

**Live monitoring.** Every real question is logged to `data/logs/queries.jsonl` with its timings, grounding score and hallucination flag. The Benchmarking & Metrics tab shows live p50/p95 latency and the latest 15 queries next to this report.

*Limits of this evaluation:* 40 synthetic questions are a small sample (one question moves a percentage by 2.5 points). The hallucination check is lexical, so it can miss paraphrased errors.

## Integration into Doc-AI

The new engine is added beside Doc-AI's existing code rather than replacing it. Login, settings and the logistics "Extract Data" feature still work as before.

**Backend (FastAPI, `backend/`).**

| Path | What it is |
| --- | --- |
| `rag/` | the engine: config, extract, chunking, models, store, retrieve, llm, pipeline, ingest |
| `api/rag_routes.py` | the HTTP routes, mounted at `/api/rag` and protected by Doc-AI's JWT login |
| `main.py` | includes the router and warms the models at startup so the first question is fast |
| `data/` | PDFs, Chroma index, chunk files, embeddings, query logs, eval set and report (git-ignored) |
| `scripts/` | download_corpus, make_scanned_sample, build_eval, run_eval |

| Route | Purpose |
| --- | --- |
| `POST /api/rag/chat` | streamed answer: retrieval event, then tokens, then a final event with citations and checks |
| `POST /api/rag/ask` | the same answer as one JSON response |
| `GET /api/rag/search` | retrieval only, no LLM |
| `POST /api/rag/ingest`, `GET /api/rag/jobs/{id}` | upload a PDF and follow its ingestion stage by stage |
| `GET /api/rag/page/{doc}/{page}.png` | the cited page as an image with the chunk highlighted |
| `GET /api/rag/status`, `/metrics`, `/eval` | corpus counts, live p50/p95 latency, the eval report |

**Settings.** Every engine setting reads an environment variable with a `RAG_` prefix (for example `RAG_CHUNK_TOKENS`). *Why the prefix:* Doc-AI's `.env` already had `CHUNK_OVERLAP=120` for its old pipeline; without a prefix the new chunker would silently have used it. All keys are listed in `backend/.env.example`.

**Frontend (React + Vite, `frontend/src/`).** Four tabs replace the old chat and benchmark tabs:

- **Corpus Q&A** (`components/RagChat.jsx`): streaming chat with clickable [S#] chips, a source list, and latency and grounding badges. Beside it, the retrieval trace: a latency waterfall and each candidate's rerank, dense and BM25 scores.
- **Ingestion Pipeline** (`components/IngestionPanel.jsx`): live counters for each stage and a per-document table of pages, OCR'd pages, chunks and timings. The left panel uploads PDFs with a progress bar per stage.
- **Benchmarking & Metrics** (`components/EvalPanel.jsx`): the offline eval report, the retriever comparison and live latency from real queries.
- **Extract Data:** the original logistics extraction, unchanged in behaviour.

The old benchmark tab showed hard-coded numbers (96.2% recall, 0.884 MRR, 98.5% faithfulness, 2.84 s) before any run. These were removed; every number on screen now comes from a measurement.

## How Claude was used to build it

Claude (Claude Code, an AI coding assistant) designed, wrote, ran and measured this system on the laptop, with the project owner making the key decisions. Claude is **not** part of the running chatbot: at runtime every model is open source and local, and no data leaves the machine.

**Why use an AI coding assistant for this.** A RAG system has many interacting parts (OCR, chunking, two indexes, a reranker, an LLM, a UI, an evaluation). Each choice is cheap to try but slow to measure by hand. Claude could write a component, run it on the real corpus, read the numbers and adjust in one loop, so choices rest on measurements rather than guesses.

**What Claude did, step by step:**

1. Read the challenge and inspected the machine: Python versions, Ollama models, RAM, missing Tesseract.
2. Chose the stack and wrote the engine, API routes, React components, scripts and docs.
3. Downloaded and checked the corpus, generated the scanned sample, and ran the 36-minute ingestion.
4. Reviewed the existing Doc-AI work and flagged three problems: a templated corpus, report figures that had not been measured on that corpus, and a keyword-only recall test.
5. Found and fixed defects by testing, for example missing spaces in OCR output, missed running headers and a settings name clash.
6. Benchmarked three model sizes, prompt sizes and thread counts, and tuned the answer time from about 31 s to 6.2 s.
7. Built and ran the evaluation, and wrote this document from the measured results.

**What the project owner decided.** To build on top of Doc-AI rather than as a separate app. To stay fully local with Ollama instead of a hosted LLM, accepting answers slightly over 5 s. Which external code to trust: a suggested "trending skills" repository was reviewed but not installed, because it only listed unvetted third-party repositories.

**Limits of this way of working.** Claude's output was checked by running it, not taken on trust; every figure here comes from a logged run. Mistakes happened and were corrected. The first version was built in a separate folder because a reply was lost, and it was then moved into Doc-AI. A person should still review the code before production use.

## How to run it

On Windows the app needs Python 3.12, Node.js and Ollama; the first full ingestion takes about 36 minutes on a laptop CPU.

1. Install the local model: `ollama pull qwen2.5:1.5b`
2. Create the backend environment, in `backend/`: `py -3.12 -m venv --system-site-packages .venv`, then `.venv\Scripts\pip install -r requirements.txt` (install CPU PyTorch first if it is missing).
3. Copy `backend/.env.example` to `backend/.env` and set `SECRET_KEY`, `ADMIN_USERNAME` and `ADMIN_PASSWORD`. Optionally add `GROQ_API_KEY` for the hosted LLM.
4. Get the corpus, in `backend/`: `.venv\Scripts\python scripts\download_corpus.py`, then `.venv\Scripts\python scripts\make_scanned_sample.py`
5. Build the index: `.venv\Scripts\python -m rag.ingest`. It is safe to re-run, because unchanged files are skipped.
6. Start the API: `start-backend.cmd` in the Doc-AI folder (port 8000). Models warm up in 40–70 s.
7. Start the UI: `start-frontend.cmd` (port 5173), then open http://localhost:5173 and log in.
8. Optional evaluation: `.venv\Scripts\python scripts\build_eval.py --n 40`, then `.venv\Scripts\python scripts\run_eval.py`. The results appear in the Benchmarking & Metrics tab.

To add a document later, drop a PDF onto the upload area in the UI. It goes through the same pipeline, with a progress bar per stage.

## Limitations and next steps

The largest gaps are full-answer latency on a CPU and the depth of local answers; both have known fixes.

| Limitation | Effect | Fix |
| --- | --- | --- |
| Local 1.5B model on a CPU | full answer about 6.7 s (p50), over the 5 s target | a GPU, or set `GROQ_API_KEY` (about 1 s generation) |
| Short local context (3 passages, ~260 tokens) | multi-part questions get partial answers | larger context once generation is faster |
| Lexical hallucination check | misses paraphrased errors and flags valid paraphrases | add a small NLI model or an LLM judge for the offline eval |
| Sideways margin text (MacKay) survives cleaning | boilerplate words in some chunks | also drop repeated blocks rotated or in the side margins |
| RapidOCR on photocopy-quality scans | occasional wrong characters | install Tesseract (picked up automatically) or use a higher DPI |
| Synthetic eval questions written by a 3B model | some questions are vague or easy | add 20–30 human-written questions with gold pages |
| One ingestion at a time | large uploads queue | a worker process for ingestion |

The engine does not yet support access control per document, which a real private corpus would need.
