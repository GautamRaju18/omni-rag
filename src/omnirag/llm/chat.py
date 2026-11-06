"""Minimal synchronous chat client for Ollama, used by the retrieval workflow (Phase 9).

Phase 10 builds the full generation engine (streaming, prompt templates, context budgets);
this covers what query transforms and the context check need:

* one ``/api/chat`` call, optionally with ``format`` = JSON schema (structured output: the
  reply always parses and enum values are respected; *content* is never trusted blindly);
* deterministic by default (``temperature 0``, fixed seed), so benchmarks are repeatable for
  one model digest;
* bounded retries on connection errors, timeouts, 429 and 5xx; one retry if a structured reply
  does not parse (e.g. cut off at ``num_predict``). Everything else raises ``LLMError``.

Sync on purpose: the workflow runs in worker threads, like the retrievers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

log = structlog.get_logger(module=__name__)


class LLMError(RuntimeError):
    """The model server failed, the model is missing, or the reply was unusable."""


@dataclass(frozen=True)
class ChatResult:
    text: str
    data: Any  # parsed JSON when a schema was given, else None
    eval_tokens: int
    prompt_tokens: int
    ms: float


class OllamaChat:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        retries: int = 2,
        backoff_s: float = 0.5,
        num_ctx: int = 8192,
        keep_alive: str = "10m",
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.retries = retries
        self.backoff_s = backoff_s
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self.stats: dict[str, float] = {"calls": 0, "seconds": 0.0, "tokens": 0}

    def chat(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        num_predict: int = 512,
        seed: int = 42,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model, "stream": False, "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx,
                        "num_predict": num_predict},
        }  # fmt: skip
        if schema is not None:
            payload["format"] = schema
        last: Exception | None = None
        for attempt in range(2 if schema is not None else 1):  # one retry for unusable JSON
            body, ms = self._post(payload)
            text = body.get("message", {}).get("content", "")
            if schema is None:
                return ChatResult(text.strip(), None, body.get("eval_count", 0),
                                  body.get("prompt_eval_count", 0), ms)  # fmt: skip
            try:
                return ChatResult(text, json.loads(text), body.get("eval_count", 0),
                                  body.get("prompt_eval_count", 0), ms)  # fmt: skip
            except json.JSONDecodeError as exc:
                last = exc
                log.warning("llm_unusable_json", attempt=attempt, model=self.model)
        raise LLMError(f"{self.model} returned unusable JSON twice: {last}")

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.post("/api/chat", json=payload)
                if response.status_code == 404:
                    raise LLMError(f"Model {self.model!r} not found (ollama pull it)")
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=response.request,
                                                response=response)  # fmt: skip
                response.raise_for_status()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt == self.retries:
                    raise LLMError(f"Ollama chat failed after {attempt + 1} tries: {exc}") from exc
                time.sleep(self.backoff_s * 2**attempt)
                continue
            ms = (time.perf_counter() - started) * 1000
            body = response.json()
            self.stats["calls"] += 1
            self.stats["seconds"] += ms / 1000
            self.stats["tokens"] += body.get("eval_count", 0)
            return body, ms
        raise LLMError("unreachable")  # pragma: no cover
