"""LLM contextual enrichment ("contextual retrieval"), an opt-in experiment.

A chunk such as "| 300 | 95.9 | 96.2 |" or "It was moved to the safety cabinet" means little on
its own. Here the local chat model reads the whole document and writes one or two sentences
situating the chunk ("Chart data from the Q1 2026 review: E-7 cell retention at 45 °C …").
That context is prepended **only to the embedding text**: ``content`` (what is cited) is
unchanged, and the context is recorded as generated, with model and prompt version.

Cost: one chat call per chunk. The document comes first in every prompt for that document,
so Ollama can reuse the cached prompt prefix between calls.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from omnirag.chunking.models import Chunk
from omnirag.chunking.tokenizer import TokenCounter
from omnirag.ingestion.models import DocumentElement, ElementType

PROMPT_VERSION = "ctx-v1"
_PROMPT = (
    "<document>\n{document}\n</document>\n\n"
    "Here is a chunk from this document:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Write one or two short sentences that situate this chunk within the document, to "
    "improve search retrieval of the chunk. Name the document's subject and what the chunk "
    "is about. Do not repeat the chunk. Answer only with the context."
)


class LLMContextualizer:
    def __init__(
        self,
        base_url: str,
        model: str,
        counter: TokenCounter,
        *,
        max_document_tokens: int = 3000,
        timeout_s: float = 180.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.counter = counter
        self.max_document_tokens = max_document_tokens
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self.stats: dict[str, float] = {"calls": 0, "seconds": 0.0}

    def document_text(self, elements: list[DocumentElement]) -> str:
        text = "\n\n".join(
            e.content for e in elements if e.content and e.element_type is not ElementType.TABLE_ROW
        )
        offsets = self.counter.offsets(text)
        if len(offsets) > self.max_document_tokens:  # keep the prompt within the context
            text = text[: offsets[self.max_document_tokens - 1][1]] + "\n[…]"
        return text

    def enrich(self, chunks: list[Chunk], elements: list[DocumentElement]) -> None:
        document = self.document_text(elements)
        for chunk in chunks:
            if not chunk.retrievable:
                continue  # parents are never embedded
            started = time.perf_counter()
            generated = self._ask(_PROMPT.format(document=document, chunk=chunk.content))
            self.stats["calls"] += 1
            self.stats["seconds"] += time.perf_counter() - started
            chunk.context = f"{chunk.context}\n{generated}" if chunk.context else generated
            chunk.metadata.update(
                context_origin="generated", context_model=self.model,
                context_prompt=PROMPT_VERSION,
                context_token_count=self.counter.count(chunk.context),
            )  # fmt: skip

    def _ask(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "5m",
            "options": {"temperature": 0, "seed": 42, "num_predict": 120},
        }
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return " ".join(response.json()["message"]["content"].split())
