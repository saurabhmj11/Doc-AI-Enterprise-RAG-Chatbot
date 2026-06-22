import numpy as np
import re
from functools import lru_cache

from config import get_settings
from core.model_loader import get_embedding_model

settings = get_settings()


class ConfidenceScorer:

    WEIGHT_RETRIEVAL = 0.30
    WEIGHT_CHUNK_AGREEMENT = 0.25
    WEIGHT_ANSWER_COVERAGE = 0.25
    WEIGHT_QA_RELEVANCE = 0.20

    def __init__(self):
        self._embedding_model = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    # =====================================================
    # EMBEDDING CACHE (MAJOR SPEED FIX)
    # =====================================================

    @lru_cache(maxsize=256)
    def _embed_cached(self, text_tuple):
        texts = list(text_tuple)
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            task_type="semantic_similarity"
        )

    # =====================================================

    def compute_confidence(self, query, answer, retrieved_chunks, context):

        retrieval = self._compute_retrieval_score(retrieved_chunks)
        agreement = self._compute_chunk_agreement(retrieved_chunks)
        coverage = self._compute_answer_coverage(answer, context)
        qa_rel = self._compute_qa_relevance(query, answer)

        confidence = (
            self.WEIGHT_RETRIEVAL * retrieval +
            self.WEIGHT_CHUNK_AGREEMENT * agreement +
            self.WEIGHT_ANSWER_COVERAGE * coverage +
            self.WEIGHT_QA_RELEVANCE * qa_rel
        )

        # --- Signal source tracking ---
        # Determine the dominant retrieval signal per top chunk
        signal_sources = []
        for c in retrieved_chunks[:3]:
            if c.get("hybrid_score") is not None:
                if c.get("bm25_score", 0) > c.get("similarity_score", 0):
                    signal_sources.append("bm25")
                else:
                    signal_sources.append("dense")
            elif c.get("bm25_score") is not None:
                signal_sources.append("bm25")
            else:
                signal_sources.append("dense")

        breakdown = {
            "retrieval_score": round(retrieval, 3),
            "chunk_agreement": round(agreement, 3),
            "answer_coverage": round(coverage, 3),
            "qa_relevance": round(qa_rel, 3),
            "signal_sources": signal_sources,    # e.g. ["dense", "bm25", "dense"]
        }

        return round(max(0.0, min(1.0, confidence)), 3), breakdown

    # =====================================================

    def _safe_cosine(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Avoid divide by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(a, b) / (norm_a * norm_b))

    # =====================================================

    def _compute_retrieval_score(self, chunks):

        if not chunks:
            return 0.0

        scores = [c.get("similarity_score", 0) for c in chunks[:3]]

        weights = [0.5, 0.3, 0.2][:len(scores)]

        return sum(s*w for s, w in zip(scores, weights)) / sum(weights)

    # =====================================================

    def _compute_chunk_agreement(self, chunks):

        if len(chunks) < 2:
            return 1.0

        texts = tuple(c.get("text","") for c in chunks[:5])

        embeddings = self._embed_cached(texts)

        sims = []

        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sims.append(self._safe_cosine(embeddings[i], embeddings[j]))

        return float(np.mean(sims)) if sims else 1.0

    # =====================================================

    def _compute_answer_coverage(self, answer, context):

        if not answer or not context:
            return 0.0

        answer_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
        context_tokens = set(re.findall(r'\b\w+\b', context.lower()))

        if len(answer_tokens) < 3:
            return 0.3

        return len(answer_tokens & context_tokens) / len(answer_tokens)

    # =====================================================

    def _compute_qa_relevance(self, query, answer):

        embeddings = self._embed_cached((query, answer))

        sim = self._safe_cosine(embeddings[0], embeddings[1])

        return max(0.0, min(1.0, sim))

    # =====================================================

    def get_confidence_level(self, score):

        if score >= settings.high_confidence_threshold:
            return "high"
        elif score >= settings.low_confidence_threshold:
            return "medium"
        return "low"


_confidence_scorer = None


def get_confidence_scorer():
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = ConfidenceScorer()
    return _confidence_scorer


