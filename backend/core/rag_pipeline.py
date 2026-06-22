import logging
from typing import Optional
import concurrent.futures
import google.generativeai as genai
import re

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


class RAGPipeline:

    def __init__(self):

        self._embedding_model = None
        self._reranker = None

        self.vector_store = get_vector_store()
        self.bm25_index = get_bm25_index()          # Sparse retrieval index
        self.fuser = get_hybrid_fuser()             # RRF / alpha-blend fuser
        self.confidence_scorer = get_confidence_scorer()
        self.guardrails = get_guardrails()
        self.privacy_layer = get_privacy_layer()

        self.llm_mode = settings.llm_mode

        if self.llm_mode == "offline":
            import ollama
            self.ollama_client = ollama.Client(host=settings.ollama_base_url)
            self.ollama_model = settings.ollama_model
            self.llm = True
            self.circuit_breaker = get_ollama_circuit_breaker()
        else:
            if settings.gemini_api_key:
                genai.configure(api_key=settings.gemini_api_key)
                self.llm = genai.GenerativeModel(settings.gemini_model)
                self.circuit_breaker = get_gemini_circuit_breaker()
            else:
                self.llm = None
                self.circuit_breaker = None

        # Reranker is now always enabled as part of the hybrid pipeline
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

    # =====================================================
    # QUERY ENHANCEMENT HELPERS
    # =====================================================

    # Logistics-domain synonym map for BM25 query expansion
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
    }

    def _expand_query(self, question: str) -> str:
        """
        Expand the query for BM25 by appending domain synonyms for any
        logistics keyword found in the question.

        Pure Python, zero external calls.  Returns the expanded query string.
        """
        q_lower = question.lower()
        extras: list[str] = []

        for keyword, synonyms in self._SYNONYMS.items():
            if keyword in q_lower:
                extras.extend(s for s in synonyms if s not in q_lower)

        if not extras:
            return question  # nothing to expand

        expanded = question + " " + " ".join(extras)
        logger.debug(f"Query expanded: '{question}' -> added {len(extras)} terms")
        return expanded

    def _hyde_embedding(self, question: str):
        """
        Hypothetical Document Embeddings (HyDE):
        1. Ask the LLM to generate a short *hypothetical* answer.
        2. Embed that hypothetical answer instead of the raw question.

        This dramatically improves dense recall for abstract questions where
        the question and the relevant document text are semantically distant.

        Falls back to plain question embedding if the LLM is unavailable.
        """
        if not self.llm:
            logger.debug("HyDE: LLM not available, falling back to plain embedding")
            return self.embedding_model.encode(question, convert_to_numpy=True)

        hyde_prompt = (
            f"Write a short, factual one-paragraph answer to the following question "
            f"as if you were reading it from a logistics document:\n\n{question}"
        )

        try:
            @api_retry(api_name="Gemini")
            def _call():
                resp = self.llm.generate_content(hyde_prompt)
                return resp.text.strip() if resp.text else None

            hypothetical = self.circuit_breaker.call(_call)

            if hypothetical:
                logger.debug(f"HyDE hypothetical: {hypothetical[:80]}...")
                return self.embedding_model.encode(hypothetical, convert_to_numpy=True)

        except Exception as exc:
            logger.warning(f"HyDE generation failed, falling back to plain query: {exc}")

        return self.embedding_model.encode(question, convert_to_numpy=True)

    # =====================================================
    # MAIN ASK FUNCTION
    # =====================================================

    def ask(self, document_ids: list[str] | str, question: str) -> dict:

        if isinstance(document_ids, str):
            document_ids = [document_ids]

        # ------------------------------------------------------------------
        # 1. Embed the query (for dense retrieval)
        #    Use HyDE if enabled — embeds a hypothetical answer for better recall
        # ------------------------------------------------------------------
        if settings.enable_hyde:
            query_vec_arr = self._hyde_embedding(question)
        else:
            query_vec_arr = self.embedding_model.encode(question, convert_to_numpy=True)

        if query_vec_arr is None:
            raise RuntimeError("Embedding failed")

        query_vec = (
            query_vec_arr.tolist()
            if hasattr(query_vec_arr, "tolist")
            else query_vec_arr
        )

        # BM25 query — optionally expanded with domain synonyms
        bm25_query = (
            self._expand_query(question)
            if settings.enable_query_expansion
            else question
        )

        # ------------------------------------------------------------------
        # 2. Parallel dense + sparse retrieval
        # ------------------------------------------------------------------
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
                    bm25_query,           # <-- expanded query goes to BM25
                    settings.top_k_retrieval,
                )
                dense_chunks = dense_future.result()
                sparse_chunks = sparse_future.result()

            # 3. Fuse results (RRF or alpha-blend)
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
            # Fallback: dense-only (original behaviour)
            chunks = self.vector_store.search_parallel(
                document_ids=document_ids,
                query_embedding=query_vec,
                top_k=settings.top_k_retrieval,
            )

        # ------------------------------------------------------------------
        # 4. Guardrail — refuse if nothing relevant was found
        # ------------------------------------------------------------------
        retrieval_check = self.guardrails.check_retrieval(chunks)

        if retrieval_check.should_refuse:
            return {
                "answer": retrieval_check.message,
                "confidence": 0.0,
                "confidence_level": "low",
                "sources": [],
                "guardrail_status": "refused",
                "guardrail_message": retrieval_check.message
            }

        # ------------------------------------------------------------------
        # 5. Neural re-ranking (Cross-Encoder) — now always on by default
        # ------------------------------------------------------------------
        if self.reranker:
            chunks = self.reranker.rerank(question, chunks, top_k=settings.top_k_rerank)

        top_chunks = chunks[:settings.top_k_rerank]
        context = self._build_context(top_chunks)

        # =====================================================
        # STRUCTURED EXTRACTION ROUTING (CRITICAL FIX)
        # =====================================================

        # Check for structured keywords
        keywords = ["shipper", "consignee", "rate", "weight", "date", "load", "shipment"]
        is_structured_query = any(k in question.lower() for k in keywords)

        if is_structured_query:
            # Try fast path first
            fast_answer = self._structured_fast_path(question, context)
            if fast_answer:
                 return {
                    "answer": fast_answer,
                    "confidence": 0.95,
                    "confidence_level": "high",
                    "sources": self._format_sources(top_chunks),
                    "guardrail_status": "allowed",
                    "guardrail_message": None
                }

        # =====================================================
        # LLM CALL
        # =====================================================

        answer = self._generate_answer(question, context)

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
            "guardrail_message": None
        }

    # =====================================================
    # CONTEXT BUILDER
    # =====================================================

    def _build_context(self, chunks):

        parts = []

        for i, chunk in enumerate(chunks, 1):
            filename = chunk.get("filename", "Unknown")
            page = chunk.get("page")
            parts.append(
                f"[Source {i} - {filename} Page {page}]:\n{chunk['text']}"
            )

        return "\n\n".join(parts)

    # =====================================================
    # STRUCTURED FAST PATH (FIXED VERSION)
    # =====================================================

    # =====================================================
    # STRUCTURED FAST PATH (FIXED VERSION)
    # =====================================================

    def _extract_pickup_block(self, context):
        # Extract pickup block safely using structural anchors
        match = re.search(
            r'Pickup\s*\n(.+?)(?=\n\s*Delivery|\n\s*Drop|\n\s*Consignee|\Z)',
            context,
            re.S | re.I
        )

        if not match:
            return None

        block = match.group(1).strip()
        lines = [l.strip() for l in block.split("\n") if l.strip()]

        # Filter out common noise
        cleaned_lines = []
        for line in lines:
            if any(x in line for x in ["S.No", "Commodity", "Weight", "Quantity", "Shipping Date"]):
                continue
            cleaned_lines.append(line)

        if cleaned_lines:
            return "\n".join(cleaned_lines)

        return None

    def _extract_delivery_block(self, context):
        # Extract delivery block safely
        match = re.search(
            r'(?:Delivery|Drop)\s*\n(.+?)(?=\n\s*Rate|\n\s*Total|\n\s*Notes|\Z)',
            context,
            re.S | re.I
        )

        if not match:
            return None

        block = match.group(1).strip()
        lines = [l.strip() for l in block.split("\n") if l.strip()]

        cleaned_lines = []
        for line in lines:
            if any(x in line for x in ["S.No", "Commodity", "Weight", "Quantity", "Delivery Date"]):
                continue
            cleaned_lines.append(line)

        if cleaned_lines:
            return "\n".join(cleaned_lines)
            
        return None

    def _structured_fast_path(self, question, context):

        # Remove source labels to prevent pollution
        context = re.sub(r'\[Source.*?\]:', '', context)
        q = question.lower()

        # 1. SHIPPER / ORIGIN
        if any(k in q for k in ["shipper", "sender", "from", "origin"]):
            val = self._extract_pickup_block(context)
            if val and len(val) < 200: # Length guard
                return f"**Shipper:**\n{val}\n\n✅ Instant answer"

        # 2. CONSIGNEE / DESTINATION
        if any(k in q for k in ["consignee", "receiver", "to", "destination"]):
            val = self._extract_delivery_block(context)
            if val and len(val) < 200:
                return f"**Consignee:**\n{val}\n\n✅ Instant answer"

        # 3. GENERIC REGEX FALLBACK FOR OTHER FIELDS
        patterns = {
            "load id": r'(?:Load|Shipment|BOL)\s*(?:ID|#)?[:\s]*([A-Z0-9-]+)',
            "rate": r'(?:Rate|Total|Amount|Charges)[:\s]*(\$?\s*[\d,]+\.?\d*)',
            "weight": r'(?:Weight|Gross)\s*[:\s]*([\d,]+\s*(?:lbs|kg)?)',
            "date": r'(?:Date|Pickup Date|Delivery Date)\s*[:\s]*(\d{2}[-/]\d{2}[-/]\d{4})'
        }

        for key, pattern in patterns.items():
            if key in q:
                match = re.search(pattern, context, re.I)
                if match:
                    val = match.group(1).strip()
                    return f"**{key.title()}:**\n{val}\n\n✅ Instant answer"

        return None

    # =====================================================
    # LLM GENERATION (SIMPLIFIED + SAFE)
    # =====================================================

    def _generate_answer(self, question, context):

        if not self.llm:
            return self._extractive_fallback(question, context)

        # Apply PII/PHI masking for production safety
        safe_question = self.privacy_layer.mask_text(question)
        safe_context = self.privacy_layer.mask_text(context)

        prompt = f"""
Answer ONLY using this context:

{safe_context}

Question: {safe_question}
"""

        try:

            @api_retry(api_name="Gemini")
            def call_llm():
                response = self.llm.generate_content(prompt)

                if not response.text:
                    raise APIError("Empty response")

                return response.text.strip()

            return self.circuit_breaker.call(call_llm)

        except Exception as e:
            logger.warning(f"LLM failed: {e}")
            return self._extractive_fallback(question, context)

    # =====================================================
    # FALLBACK
    # =====================================================

    def _extractive_fallback(self, question, context):
        
        # Clean context to just text
        clean_text = re.sub(r'\[Source.*?\]:', '', context).strip()
        
        # Split by double newline to get first paragraph/header block
        first_block = clean_text.split("\n\n")[0]
        
        # Strict length limit to avoid dumping entire doc
        if len(first_block) > 300:
            first_block = first_block[:300] + "..."
            
        return f"Based on document:\n{first_block}"

    # =====================================================
    # SOURCE FORMATTER
    # =====================================================

    def _format_sources(self, chunks):

        sources = []

        for chunk in chunks:

            text = chunk["text"]

            if len(text) > 300:
                text = text[:300] + "..."

            sources.append({
                "text": text,
                "page": chunk.get("page"),
                "filename": chunk.get("filename"),
                "chunk_id": chunk.get("chunk_id"),
                "similarity_score": round(chunk.get("similarity_score", 0), 3)
            })

        return sources


_rag_pipeline = None

def get_rag_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline
