"""The pinned bge-m3 tokenizer counts exactly what Ollama's bge-m3 sees (run: -m integration)."""

from __future__ import annotations

import httpx
import pytest

from omnirag.chunking import load_tokenizer
from omnirag.core.config import Settings

pytestmark = pytest.mark.integration

TEXTS = [
    "Cell AUR-C103 failed at cycle 412 because of gas evolution at 45 °C.",
    "Na₂Fe[Fe(CN)₆]",
    "Cell: AUR-C103 | Electrolyte: E-5 | Temp (°C): 45 | Cycles: 412 | Retention (%): 72.5",
]


def test_token_counts_match_ollama_bge_m3() -> None:
    settings = Settings()
    tokenizer = load_tokenizer(settings)
    for text in TEXTS:
        payload = {"model": settings.embedding_model, "input": text}
        body = httpx.post(f"{settings.ollama_base_url}/api/embed", json=payload, timeout=120).json()
        # Ollama counts the 2 special tokens (<s> ... </s>); chunk sizes exclude them.
        assert tokenizer.count(text) + 2 == body["prompt_eval_count"], text
