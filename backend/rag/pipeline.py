"""End-to-end RAG answer: retrieve -> build cited context -> stream LLM answer -> verify citations/grounding."""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Iterator

from . import config, llm
from .retrieve import retrieve
from .store import lex_tokens

NOT_FOUND = "I could not find this in the indexed documents."

# Kept short on purpose: on a CPU every prompt token costs ~6 ms before the first answer token.
SYSTEM_PROMPT = """Answer the question using only the numbered sources. Write at most {words} words. \
End every sentence with the tag of the source it comes from, like [S1]. \
If the sources do not answer the question, reply exactly: "{not_found}\""""


def system_prompt() -> str:
    words = config.ANSWER_WORDS_LOCAL if llm.is_local() else config.ANSWER_WORDS_HOSTED
    return SYSTEM_PROMPT.format(words=words, not_found=NOT_FOUND)

_SENT = re.compile(r"(?<=[.!?])\s+")
_TAG = re.compile(r"\[S(\d+)\]")
_DUP_TAG = re.compile(r"(\[S\d+\])(?:\s*\1)+")
_log_lock = threading.Lock()


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35) + 1


def _compress(text: str, query: str, budget: int) -> str:
    """Keep the sentences most lexically related to the query (in original order) within `budget` tokens."""
    if _approx_tokens(text) <= budget:
        return text
    q = set(lex_tokens(query))
    sents = [s for s in _SENT.split(text) if s.strip()]
    scored = sorted(range(len(sents)), key=lambda i: -len(q & set(lex_tokens(sents[i]))))
    keep, used = set(), 0
    for i in scored:
        n = _approx_tokens(sents[i])
        if used + n > budget:
            continue
        keep.add(i)
        used += n
    # also keep immediate neighbours of the best sentence for coherence when room remains
    return " ... ".join(sents[i] for i in sorted(keep)) or " ".join(text.split()[: int(budget / 1.35)])


def _page_label(c: dict) -> str:
    return f"p. {c['page_start']}" if c["page_start"] == c["page_end"] else f"pp. {c['page_start']}-{c['page_end']}"


def build_context(query: str, results: list[dict]) -> tuple[str, list[dict]]:
    budget = llm.context_budget()
    n = min(config.CONTEXT_CHUNKS_LOCAL, len(results)) if llm.is_local() else len(results)
    used = results[:n]
    per = budget // max(n, 1)
    sources, parts = [], []
    for i, c in enumerate(used, 1):
        text = _compress(c["text"], query, per)
        sources.append({"tag": f"S{i}", "chunk_id": c["chunk_id"], "filename": c["filename"],
                        "doc_id": c["doc_id"], "pages": c["pages"], "page_label": _page_label(c),
                        "bboxes": c["bboxes"], "text": text})
        # A local model only needs the tag (filename + page are attached to citations afterwards, saving
        # ~20 prompt tokens per source); hosted models also get the label for richer answers.
        label = f"[S{i}]" if llm.is_local() else f"[S{i}] ({c['filename']}, {_page_label(c)})"
        parts.append(f"{label}\n{text}")
    return "\n\n".join(parts), sources


def verify(answer: str, sources: list[dict]) -> dict:
    """Citation validity + a lexical grounding check used as a cheap hallucination signal."""
    tags = [int(t) for t in _TAG.findall(answer)]
    valid = [t for t in tags if 1 <= t <= len(sources)]
    cited = sorted(set(valid))
    refused = NOT_FOUND.lower()[:30] in answer.lower()
    ctx_tokens = {i + 1: set(lex_tokens(s["text"])) for i, s in enumerate(sources)}
    all_ctx = set().union(*ctx_tokens.values()) if ctx_tokens else set()
    sentences, unsupported, uncited = [], 0, 0
    for sent in _SENT.split(answer):
        stags = [int(t) for t in _TAG.findall(sent)]
        toks = set(lex_tokens(_TAG.sub("", sent)))
        if len(toks) < 4:
            continue
        ref = set().union(*(ctx_tokens.get(t, set()) for t in stags)) if stags else all_ctx
        support = len(toks & ref) / len(toks)
        ok = support >= 0.5
        unsupported += not ok
        uncited += not stags
        sentences.append({"text": sent[:200], "support": round(support, 2), "cited": bool(stags), "supported": ok})
    n = len(sentences)
    return {
        "refused": refused,
        "citations_total": len(tags),
        "citations_valid": len(valid),
        "citation_validity": round(len(valid) / len(tags), 3) if tags else (1.0 if refused else 0.0),
        "cited_sources": cited,
        "sentences": sentences,
        "unsupported_sentences": unsupported,
        "uncited_sentences": uncited,
        "grounding_score": round(1 - unsupported / n, 3) if n else (1.0 if refused else 0.0),
        "hallucination_flag": (not refused) and n > 0 and unsupported / n > 0.34,
    }


def attribute_citations(answer: str, sources: list[dict]) -> tuple[str, int]:
    """Small local models often drop [S#] tags. Tag each untagged factual sentence with the source that
    lexically supports it best (>= 50% of its content words). Returns (answer, tags_added)."""
    ctx_tokens = [set(lex_tokens(s["text"])) for s in sources]
    added = 0
    lines = []
    for line in answer.split("\n"):
        sents = []
        for sent in _SENT.split(line):
            toks = set(lex_tokens(sent))
            if not _TAG.search(sent) and len(toks) >= 4 and ctx_tokens:
                support = [len(toks & c) / len(toks) for c in ctx_tokens]
                best = max(range(len(support)), key=support.__getitem__)
                if support[best] >= 0.5:
                    sent = f"{sent.rstrip()} [S{best + 1}]"
                    added += 1
            sents.append(sent)
        lines.append(" ".join(sents))
    # a tag the model put after the full stop is split off as its own fragment: collapse "[S1] [S1]"
    return _DUP_TAG.sub(r"\1", "\n".join(lines)), added


def extractive_answer(question: str, sources: list[dict], n: int = 3) -> str:
    """No-LLM answer: the sentences that best match the question, each tagged with its source."""
    q = set(lex_tokens(question))
    scored = []
    for i, s in enumerate(sources, 1):
        for sent in _SENT.split(s["text"]):
            toks = set(lex_tokens(sent))
            if len(toks) >= 5:
                # earlier (higher-ranked) sources win ties
                scored.append((len(q & toks) / (len(q) or 1) - i * 0.01, sent.strip(), i))
    best = sorted(scored, key=lambda x: -x[0])[:n]
    if not best or best[0][0] <= 0:
        return NOT_FOUND
    return "Most relevant passages (extractive, LLM offline):\n" + "\n".join(f"- {s} [S{i}]" for _, s, i in best)


def answer_stream(question: str, history: list[dict] | None = None) -> Iterator[dict]:
    t_start = time.perf_counter()
    ret = retrieve(question)
    context, sources = build_context(question, ret["results"])
    t_ret = time.perf_counter()
    yield {"type": "retrieval", "candidates": ret["candidates"], "counts": ret["counts"],
           "timings": ret["timings"], "sources": sources, "provider": llm.provider(), "model": llm.model_name()}

    messages = [{"role": "system", "content": system_prompt()}]
    # Chat history costs prompt tokens; a local CPU model only gets the previous question for context.
    history = history or []
    for h in [h for h in history if h["role"] == "user"][-1:] if llm.is_local() else history[-4:]:
        messages.append({"role": h["role"], "content": h["content"][:600]})
    messages.append({"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"})
    parts, t_first = [], None
    try:
        for piece in llm.stream_chat(messages):
            if t_first is None:
                t_first = time.perf_counter()
            parts.append(piece)
            yield {"type": "token", "text": piece}
    except Exception as e:
        # LLM unreachable (e.g. Ollama stopped, no Groq key): fall back to an extractive, still-cited answer
        if not parts:
            fallback = extractive_answer(question, sources)
            t_first = time.perf_counter()
            parts.append(fallback)
            yield {"type": "notice", "message": f"LLM unavailable ({type(e).__name__}); showing extractive answer"}
            yield {"type": "token", "text": fallback}
        else:
            yield {"type": "error", "message": str(e)}
    raw = "".join(parts).strip()
    t_end = time.perf_counter()

    model_check = verify(raw, sources)
    answer, auto_tags = attribute_citations(raw, sources)
    check = verify(answer, sources)
    # keep the model's own citation behaviour visible next to the post-processed result
    check.update(model_citations_total=model_check["citations_total"],
                 model_citation_validity=model_check["citation_validity"], auto_citations=auto_tags)
    citations = [{k: sources[i - 1][k] for k in ("tag", "filename", "doc_id", "pages", "page_label", "chunk_id", "bboxes")}
                 for i in check["cited_sources"]]
    timings = {
        **ret["timings"],
        "context_ms": round((t_ret - t_start) * 1000 - ret["timings"]["retrieval_ms"], 1),
        "ttft_ms": round(((t_first or t_end) - t_start) * 1000, 1),
        "generation_ms": round((t_end - t_ret) * 1000, 1),
        "total_ms": round((t_end - t_start) * 1000, 1),
    }
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "question": question, "provider": llm.provider(),
              "model": llm.model_name(), "timings": timings, "citations": len(citations),
              "grounding_score": check["grounding_score"], "hallucination_flag": check["hallucination_flag"],
              "refused": check["refused"]}
    with _log_lock, open(config.LOG_DIR / "queries.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    yield {"type": "done", "answer": answer, "citations": citations, "verification": check, "timings": timings}


def answer(question: str, history: list[dict] | None = None) -> dict:
    out = {}
    for ev in answer_stream(question, history):
        if ev["type"] == "retrieval":
            out["retrieval"] = ev
        elif ev["type"] in ("done", "error"):
            out.update(ev)
    return out
