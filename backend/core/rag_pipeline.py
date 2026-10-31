"""
Production RAG Pipeline Module
===============================
- Multi-document retrieval with dense vector + BM25 sparse hybrid fusion (RRF)
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- Strict source citation: [Filename, Page X]
- Fallback response: "Information not found in the provided documents."
- Detailed latency telemetry with p50, p95, p99 percentiles (target: 2–5s)
- PII/PHI privacy protection masking
- Circuit breaker resilience for LLM calls
"""

import logging
import time
import re
import concurrent.futures
from typing import Optional

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None

from config import get_settings
from core.model_loader import get_embedding_model
from core.vector_store import get_vector_store
from core.bm25_index import get_bm25_index
from core.hybrid_fuser import get_hybrid_fuser
from core.confidence_scorer import get_confidence_scorer
from core.guardrails import get_guardrails
from core.reranker import get_reranker
from core.privacy import get_privacy_layer
from core.error_handling import (
    get_gemini_circuit_breaker,
    get_ollama_circuit_breaker,
    api_retry,
    APIError
)

settings = get_settings()
logger = logging.getLogger(__name__)


# ============================================================
# LATENCY TRACKER
# ============================================================

class LatencyTracker:
    """Tracks query latency breakdown and computes p50, p95, p99 metrics."""

    def __init__(self, maxlen: int = 1000):
        self.records: list[dict] = []
        self.maxlen = maxlen

    def add(self, total_ms: float, retrieval_ms: float, rerank_ms: float, generation_ms: float):
        self.records.append({
            "timestamp": time.time(),
            "total_ms": total_ms,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": generation_ms,
        })
        if len(self.records) > self.maxlen:
            self.records.pop(0)

    def get_stats(self) -> dict:
        if not self.records:
            return {
                "total_queries": 0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "avg_ms": 0.0,
                "avg_retrieval_ms": 0.0,
                "avg_rerank_ms": 0.0,
                "avg_generation_ms": 0.0,
            }

        totals = sorted(r["total_ms"] for r in self.records)
        n = len(totals)

        def percentile(p: float) -> float:
            idx = int(n * p)
            return round(totals[min(idx, n - 1)], 1)

        return {
            "total_queries": n,
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "avg_ms": round(sum(totals) / n, 1),
            "avg_retrieval_ms": round(sum(r["retrieval_ms"] for r in self.records) / n, 1),
            "avg_rerank_ms": round(sum(r["rerank_ms"] for r in self.records) / n, 1),
            "avg_generation_ms": round(sum(r["generation_ms"] for r in self.records) / n, 1),
        }


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:

    def __init__(self):
        self._embedding_model = None
        self._reranker = None

        self.vector_store = get_vector_store()
        self.bm25_index = get_bm25_index()
        self.fuser = get_hybrid_fuser()
        self.confidence_scorer = get_confidence_scorer()
        self.guardrails = get_guardrails()
        self.privacy_layer = get_privacy_layer()
        self.latency_tracker = LatencyTracker()

        self.llm_mode = settings.llm_mode

        if self.llm_mode == "offline":
            try:
                import ollama
                self.ollama_client = ollama.Client(host=settings.ollama_base_url)
                self.ollama_model = settings.ollama_model
                self.llm = True
                self.circuit_breaker = get_ollama_circuit_breaker()
            except Exception as e:
                logger.warning(f"Ollama initialization failed: {e}")
                self.llm = False
                self.circuit_breaker = None
        else:
            if settings.gemini_api_key and genai is not None:
                genai.configure(api_key=settings.gemini_api_key)
                self.llm = genai.GenerativeModel(settings.gemini_model)
                self.circuit_breaker = get_gemini_circuit_breaker()
            else:
                self.llm = None
                self.circuit_breaker = None

        if settings.enable_reranking:
            self._reranker = get_reranker()
        else:
            self._reranker = None

    @property
    def reranker(self):
        return self._reranker

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    # Domain synonyms for BM25 query expansion
    _SYNONYMS: dict[str, list[str]] = {
        "shipper":     ["sender", "origin", "from", "pickup"],
        "consignee":   ["receiver", "destination", "to", "delivery", "recipient"],
        "rate":        ["price", "charge", "cost", "amount", "total", "fee"],
        "weight":      ["gross", "lbs", "kg", "mass"],
        "date":        ["datetime", "time", "schedule", "when"],
        "load":        ["shipment", "bol", "bill of lading", "freight"],
        "carrier":     ["trucker", "driver", "transport", "hauler"],
        "equipment":   ["trailer", "truck", "container", "van", "flatbed"],
        "pickup":      ["collection", "origin stop", "loading"],
        "delivery":    ["drop", "unloading", "consignee stop"],
        "commodity":   ["goods", "cargo", "freight", "product", "item"],
        "mode":        ["transport mode", "method", "ftl", "ltl"],
        "revenue":     ["income", "earnings", "sales", "turnover"],
        "profit":      ["margin", "net income", "operating income"],
        "liability":   ["debt", "obligation", "payable"],
        "regulation":  ["compliance", "rule", "standard", "policy"],
    }

    def _expand_query(self, question: str) -> str:
        """Expand query with relevant synonyms for BM25 sparse search."""
        q_lower = question.lower()
        extras: list[str] = []

        for keyword, synonyms in self._SYNONYMS.items():
            if keyword in q_lower:
                extras.extend(s for s in synonyms if s not in q_lower)

        if not extras:
            return question

        expanded = question + " " + " ".join(extras)
        logger.debug(f"Query expanded: '{question}' -> added {len(extras)} terms")
        return expanded

    def _hyde_embedding(self, question: str):
        """Hypothetical Document Embeddings (HyDE) for complex conceptual queries."""
        if not self.llm or self.llm_mode == "offline":
            return self.embedding_model.encode(question, convert_to_numpy=True)

        hyde_prompt = (
            f"Write a short, factual one-paragraph answer to the following question "
            f"based on technical and business documents:\n\n{question}"
        )

        try:
            @api_retry(api_name="Gemini")
            def _call():
                resp = self.llm.generate_content(hyde_prompt)
                return resp.text.strip() if resp.text else None

            hypothetical = self.circuit_breaker.call(_call)
            if hypothetical:
                return self.embedding_model.encode(hypothetical, convert_to_numpy=True)
        except Exception as exc:
            logger.warning(f"HyDE generation failed, falling back to plain query: {exc}")

        return self.embedding_model.encode(question, convert_to_numpy=True)

    # ============================================================
    # MAIN PIPELINE EXECUTION
    # ============================================================

    def ask(self, document_ids: list[str] | str, question: str) -> dict:
        """
        Execute end-to-end RAG pipeline with latency monitoring.
        Target round-trip latency: 2–5 seconds.
        """
        t_start = time.time()

        if isinstance(document_ids, str):
            document_ids = [document_ids]

        # If empty or ["all"], resolve to all documents in vector store
        if not document_ids or document_ids == ["all"]:
            all_docs = self.vector_store.list_documents()
            document_ids = [d["document_id"] for d in all_docs]

        # -------------------------------------------------------------
        # Phase 1: Query Embedding & Retrieval (Dense + Sparse Fusion)
        # -------------------------------------------------------------
        t_retrieval_start = time.time()

        if settings.enable_hyde:
            query_vec_arr = self._hyde_embedding(question)
        else:
            query_vec_arr = self.embedding_model.encode(question, convert_to_numpy=True)

        if query_vec_arr is None:
            raise RuntimeError("Embedding generation failed")

        query_vec = (
            query_vec_arr.tolist()
            if hasattr(query_vec_arr, "tolist")
            else query_vec_arr
        )

        bm25_query = (
            self._expand_query(question)
            if settings.enable_query_expansion
            else question
        )

        if settings.enable_hybrid_search:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(
                    self.vector_store.search_parallel,
                    document_ids,
                    query_vec,
                    settings.top_k_retrieval,
                )
                sparse_future = executor.submit(
                    self.bm25_index.search_parallel,
                    document_ids,
                    bm25_query,
                    settings.top_k_retrieval,
                )
                dense_chunks = dense_future.result()
                sparse_chunks = sparse_future.result()

            chunks = self.fuser.fuse(
                dense_chunks,
                sparse_chunks,
                top_k=settings.top_k_retrieval,
            )
            logger.info(
                f"Hybrid retrieval: {len(dense_chunks)} dense + "
                f"{len(sparse_chunks)} sparse → {len(chunks)} fused chunks"
            )
        else:
            chunks = self.vector_store.search_parallel(
                document_ids=document_ids,
                query_embedding=query_vec,
                top_k=settings.top_k_retrieval,
            )

        t_retrieval_ms = round((time.time() - t_retrieval_start) * 1000, 1)

        # -------------------------------------------------------------
        # Guardrails: refusal if nothing relevant found
        # -------------------------------------------------------------
        retrieval_check = self.guardrails.check_retrieval(chunks)
        if retrieval_check.should_refuse:
            t_total_ms = round((time.time() - t_start) * 1000, 1)
            self.latency_tracker.add(t_total_ms, t_retrieval_ms, 0.0, 0.0)
            return {
                "answer": "Information not found in the provided documents.",
                "confidence": 0.0,
                "confidence_level": "low",
                "sources": [],
                "guardrail_status": "refused",
                "guardrail_message": retrieval_check.message,
                "latency": {
                    "total_ms": t_total_ms,
                    "retrieval_ms": t_retrieval_ms,
                    "rerank_ms": 0.0,
                    "generation_ms": 0.0
                }
            }

        # -------------------------------------------------------------
        # Phase 2: Neural Cross-Encoder Re-ranking
        # -------------------------------------------------------------
        t_rerank_start = time.time()
        if self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_k=settings.top_k_rerank)

        top_chunks = chunks[:settings.top_k_rerank]
        t_rerank_ms = round((time.time() - t_rerank_start) * 1000, 1)

        context = self._build_context(top_chunks)

        # -------------------------------------------------------------
        # Phase 3: Answer Generation (LLM or Extractive Fallback)
        # -------------------------------------------------------------
        t_gen_start = time.time()
        answer = self._generate_answer(question, context)
        t_gen_ms = round((time.time() - t_gen_start) * 1000, 1)

        t_total_ms = round((time.time() - t_start) * 1000, 1)
        self.latency_tracker.add(t_total_ms, t_retrieval_ms, t_rerank_ms, t_gen_ms)

        confidence, breakdown = self.confidence_scorer.compute_confidence(
            question,
            answer,
            top_chunks,
            context
        )
        confidence_level = self.confidence_scorer.get_confidence_level(confidence)

        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "sources": self._format_sources(top_chunks),
            "guardrail_status": "allowed",
            "guardrail_message": None,
            "latency": {
                "total_ms": t_total_ms,
                "retrieval_ms": t_retrieval_ms,
                "rerank_ms": t_rerank_ms,
                "generation_ms": t_gen_ms
            }
        }

    # ============================================================
    # CONTEXT BUILDER & SOURCE CITATION
    # ============================================================

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get("filename", "Document")
            page = chunk.get("page", 1)
            parts.append(
                f"[Source {i} - {filename}, Page {page}]:\n{chunk['text']}"
            )
        return "\n\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer from context with strict source citations.
        If information is missing, explicitly state:
        'Information not found in the provided documents.'
        """
        safe_question = self.privacy_layer.mask_text(question)
        safe_context = self.privacy_layer.mask_text(context)

        prompt = (
            "You are an expert document research assistant.\n"
            "Answer the user's question accurately and concisely using ONLY the provided context excerpts.\n\n"
            "STRICT RULES:\n"
            "1. Every factual statement must cite the source document and page number in square brackets, e.g., [document.pdf, Page 12].\n"
            "2. If the context does not contain sufficient facts to answer the question, you MUST reply ONLY:\n"
            "   \"Information not found in the provided documents.\"\n"
            "3. Do not assume, speculate, or extrapolate.\n\n"
            f"CONTEXT:\n{safe_context}\n\n"
            f"QUESTION: {safe_question}\n\n"
            "ANSWER:"
        )

        # OFFLINE LLM (Ollama)
        if self.llm_mode == "offline" and getattr(self, "ollama_client", None):
            try:
                @api_retry(api_name="Ollama")
                def call_ollama():
                    resp = self.ollama_client.generate(
                        model=self.ollama_model,
                        prompt=prompt
                    )
                    return resp.get("response", "").strip()

                return self.circuit_breaker.call(call_ollama)
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}")
                return self._extractive_fallback(question, context)

        # ONLINE LLM (Gemini)
        elif self.llm and self.circuit_breaker:
            try:
                @api_retry(api_name="Gemini")
                def call_gemini():
                    resp = self.llm.generate_content(prompt)
                    if not resp.text:
                        raise APIError("Empty response from LLM")
                    return resp.text.strip()

                return self.circuit_breaker.call(call_gemini)
            except Exception as e:
                logger.warning(f"Gemini LLM failed: {e}")
                return self._extractive_fallback(question, context)

        # EXTRACTIVE FALLBACK (No LLM configured or offline fallback)
        return self._extractive_fallback(question, context)

    def _extractive_fallback(self, question: str, context: str) -> str:
        """
        High-precision extractive fallback when no LLM is running.
        Selects the most relevant paragraph from top retrieved chunks and attaches citations.
        """
        if not context.strip():
            return "Information not found in the provided documents."

        # Parse source blocks: [Source X - filename, Page Y]: text
        blocks = re.findall(
            r'\[Source \d+ - ([^\]]+)\]:\n(.*?)(?=\n\n\[Source|\Z)',
            context,
            re.S
        )

        if not blocks:
            return "Information not found in the provided documents."

        q_terms = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', question.lower())) - {
            "what", "when", "where", "which", "whose", "why", "from", "that", "this", "have", "with"
        }

        best_passage = None
        best_source = None
        best_score = 0

        for source_tag, block_text in blocks:
            paragraphs = [p.strip() for p in block_text.split("\n\n") if p.strip()]
            for para in paragraphs:
                para_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', para.lower()))
                overlap = len(q_terms & para_words)
                if overlap > best_score:
                    best_score = overlap
                    best_passage = para
                    best_source = source_tag

        if best_passage and best_score >= 1:
            snippet = best_passage[:400].strip()
            if len(best_passage) > 400:
                snippet += "..."
            return f"{snippet} [{best_source}]"

        return "Information not found in the provided documents."

    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        sources = []
        for chunk in chunks:
            text = chunk["text"]
            if len(text) > 350:
                text = text[:350] + "..."

            sources.append({
                "text": text,
                "page": chunk.get("page", 1),
                "filename": chunk.get("filename", "unknown"),
                "chunk_id": chunk.get("chunk_id", ""),
                "similarity_score": round(chunk.get("similarity_score", 0), 3),
                "hybrid_score": round(chunk.get("hybrid_score", chunk.get("similarity_score", 0)), 3),
                "signal_source": chunk.get("signal_source", "hybrid")
            })
        return sources

    def get_latency_stats(self) -> dict:
        """Return p50, p95, p99 and average latency metrics."""
        return self.latency_tracker.get_stats()


# ============================================================
# SINGLETON INSTANCE
# ============================================================

_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
