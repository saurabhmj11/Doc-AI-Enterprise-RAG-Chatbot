"""Streaming chat completion against Groq (hosted open-source Llama) or a local Ollama model."""
from __future__ import annotations

import json
from typing import Iterator

import httpx

from . import config

_client = httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0))


def provider() -> str:
    if config.LLM_PROVIDER != "auto":
        return config.LLM_PROVIDER
    return "groq" if config.GROQ_API_KEY else "ollama"


def model_name() -> str:
    return config.GROQ_MODEL if provider() == "groq" else config.OLLAMA_MODEL


def is_local() -> bool:
    return provider() != "groq"


def context_budget() -> int:
    return config.CONTEXT_TOKENS_LOCAL if is_local() else config.CONTEXT_TOKENS_HOSTED


def max_tokens_default() -> int:
    return config.LLM_MAX_TOKENS_LOCAL if is_local() else config.LLM_MAX_TOKENS_HOSTED


def stream_chat(messages: list[dict], max_tokens: int | None = None) -> Iterator[str]:
    max_tokens = max_tokens or max_tokens_default()
    if provider() == "groq":
        body = {"model": config.GROQ_MODEL, "messages": messages, "stream": True,
                "temperature": config.LLM_TEMPERATURE, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
        with _client.stream("POST", "https://api.groq.com/openai/v1/chat/completions",
                            json=body, headers=headers) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Groq error {r.status_code}: {r.read().decode()[:300]}")
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
    else:
        # num_ctx must stay identical across calls: changing it forces Ollama to reload the model
        options = {"temperature": config.LLM_TEMPERATURE, "num_predict": max_tokens, "num_ctx": 2048}
        if config.OLLAMA_THREADS:
            options["num_thread"] = config.OLLAMA_THREADS
        body = {"model": config.OLLAMA_MODEL, "messages": messages, "stream": True, "keep_alive": "60m",
                "options": options}
        with _client.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Ollama error {r.status_code}: {r.read().decode()[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                piece = msg.get("message", {}).get("content")
                if piece:
                    yield piece
                if msg.get("done"):
                    break


def complete(messages: list[dict], max_tokens: int | None = None) -> str:
    return "".join(stream_chat(messages, max_tokens))


def warmup() -> None:
    """Load the local model into memory so the first user query is not penalised."""
    if provider() == "ollama":
        try:
            complete([{"role": "user", "content": "hi"}], max_tokens=1)
        except Exception:
            pass
