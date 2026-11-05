"""OllamaEmbedder with a fake Ollama: batching, order, normalisation, cache, errors."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from omnirag.embeddings import EmbeddingCache, EmbeddingError, OllamaEmbedder


class FakeOllama:
    def __init__(self, status: int = 200) -> None:
        self.batches: list[list[str]] = []
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        self.batches.append(texts)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        # unnormalised, text-dependent vectors
        return httpx.Response(200, json={"embeddings": [[len(t), 1.0, 0.0] for t in texts]})


def _embedder(fake, tmp_path: Path | None = None, batch_size: int = 2) -> OllamaEmbedder:
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(fake))
    cache = EmbeddingCache(tmp_path / "e.sqlite") if tmp_path else None
    return OllamaEmbedder("http://ollama", "m", batch_size=batch_size, cache=cache, client=client)


def test_batches_order_and_normalisation() -> None:
    fake = FakeOllama()
    vectors = _embedder(fake).embed(["a", "bbb", "cc", "a"])
    assert fake.batches == [["a", "bbb"], ["cc"]]  # duplicates embedded once, in batches of 2
    assert vectors.shape == (4, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], vectors[3]) and not np.allclose(vectors[0], vectors[1])


def test_cache_survives_new_embedder(tmp_path: Path) -> None:
    first = FakeOllama()
    before = _embedder(first, tmp_path).embed(["x", "y"])
    second = FakeOllama()
    again = _embedder(second, tmp_path)
    after = again.embed(["y", "x", "z"])
    assert second.batches == [["z"]]  # only the new text hits Ollama
    assert np.allclose(after[0], before[1]) and np.allclose(after[1], before[0])
    assert again.stats["cached"] == 2 and again.stats["embedded"] == 1


def test_http_error_raises_embedding_error() -> None:
    with pytest.raises(EmbeddingError):
        _embedder(FakeOllama(status=404)).embed(["a"])
