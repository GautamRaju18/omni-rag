"""OllamaDenseEmbedder against a fake Ollama: versioning, validation, errors, cache, refs."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from omnirag.chunking.models import Chunk, ChunkingStrategy
from omnirag.embeddings import (
    EmbeddingCache,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingSpace,
    OllamaDenseEmbedder,
)
from omnirag.embeddings.base import sparse_dot, validate_dense
from omnirag.embeddings.service import ChunkEmbedder, load_vectors

DIGEST = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"


class FakeOllama:
    def __init__(self, *, dim: int = 3, fail_times: int = 0, reject: str | None = None,
                 vector_dim: int | None = None, digest: str = DIGEST) -> None:  # fmt: skip
        self.dim, self.digest = dim, digest
        self.vector_dim = vector_dim or dim
        self.fail_times, self.reject = fail_times, reject
        self.embed_batches: list[list[str]] = []
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "m:latest", "digest": self.digest}]}
            )
        if request.url.path == "/api/show":
            info = {"general.architecture": "bert", "bert.embedding_length": self.dim}
            return httpx.Response(200, json={"model_info": info})
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail_times:
            self.fail_times -= 1
            return httpx.Response(503, json={"error": "loading model"})
        if self.reject is not None and self.reject in body["input"]:
            return httpx.Response(
                413, json={"error": "the input length exceeds the context length"}
            )
        self.embed_batches.append(body["input"])
        pad = [0.0] * (self.vector_dim - 2)
        vectors = [[float(len(t)), 1.0, *pad] for t in body["input"]]
        return httpx.Response(200, json={"embeddings": vectors})


def _embedder(fake: FakeOllama, tmp_path: Path | None = None, **kw) -> OllamaDenseEmbedder:
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(fake))
    cache = EmbeddingCache(tmp_path / "e.sqlite") if tmp_path else None
    return OllamaDenseEmbedder("http://ollama", "m", batch_size=2, cache=cache, client=client,
                               backoff_s=0, **kw)  # fmt: skip


def test_space_is_versioned_by_digest_and_validated() -> None:
    space = _embedder(FakeOllama(), expected_dim=3).space
    assert space == EmbeddingSpace(provider="ollama", model="m", revision=DIGEST, dim=3,
                                   max_tokens=2048)  # fmt: skip
    assert space.space_id == "dense-ollama-m-790764642607"
    assert _embedder(FakeOllama(digest="f" * 64)).space.space_id == "dense-ollama-m-ffffffffffff"
    with pytest.raises(EmbeddingError, match="dimension 3, expected 1024"):
        _ = _embedder(FakeOllama(), expected_dim=1024).space


def test_documents_and_queries_get_their_prefixes_and_are_normalised() -> None:
    fake = FakeOllama()
    e = _embedder(fake, query_prefix="q: ", document_prefix="d: ")
    docs = e.embed_documents(["aa", "bbb", "aa"])
    query = e.embed_query("x")
    assert fake.embed_batches == [["d: aa", "d: bbb"], ["q: x"]]  # duplicates once, batches of 2
    assert all(b["truncate"] is False for b in fake.bodies)  # never let Ollama cut input
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0) and np.allclose(docs[0], docs[2])
    assert query.shape == (3,)


def test_wrong_vector_size_from_server_is_rejected() -> None:
    with pytest.raises(EmbeddingError, match="expected 1×3"):
        _embedder(FakeOllama(dim=3, vector_dim=4)).embed_documents(["a"])


def test_empty_and_too_long_inputs_fail_before_the_server() -> None:
    fake = FakeOllama()
    e = _embedder(fake, max_tokens=5, count_tokens=lambda t: len(t.split()))
    with pytest.raises(EmbeddingInputError, match="Input 1 is empty") as empty:
        e.embed_documents(["ok", "  "])
    assert empty.value.index == 1
    with pytest.raises(EmbeddingInputError, match="Input 0 has 6 tokens") as long:
        e.embed_documents(["one two three four"])  # 4 + 2 special tokens > 5
    assert long.value.index == 0 and fake.embed_batches == []


def test_server_rejection_is_bisected_to_the_right_input() -> None:
    fake = FakeOllama(reject="bad")
    with pytest.raises(EmbeddingInputError, match="Input 2") as exc:
        _embedder(fake).embed_documents(["a", "b", "bad", "c"])
    assert exc.value.index == 2


def test_transient_errors_are_retried_then_reported() -> None:
    fake = FakeOllama(fail_times=2)
    e = _embedder(fake, retries=3)
    assert e.embed_documents(["a"]).shape == (1, 3) and e.stats["retries"] == 2
    with pytest.raises(EmbeddingError, match="HTTP 503"):
        _embedder(FakeOllama(fail_times=5), retries=2).embed_documents(["a"])


def test_missing_model_and_unreachable_server() -> None:
    def no_models(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(no_models))
    with pytest.raises(EmbeddingError, match="not found in Ollama"):
        _ = OllamaDenseEmbedder("http://ollama", "m", client=client).space

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(down))
    with pytest.raises(EmbeddingError, match="unreachable"):
        _ = OllamaDenseEmbedder("http://ollama", "m", client=client, retries=1, backoff_s=0).space


def test_cache_is_keyed_by_space(tmp_path: Path) -> None:
    first = FakeOllama()
    _embedder(first, tmp_path).embed_documents(["x", "y"])
    same = FakeOllama()
    _embedder(same, tmp_path).embed_documents(["y", "x", "z"])
    assert same.embed_batches == [["z"]]  # only the new text hits Ollama
    repulled = FakeOllama(digest="a" * 64)  # same name, different weights
    _embedder(repulled, tmp_path).embed_documents(["x"])
    assert repulled.embed_batches == [["x"]]  # never served a stale vector


def _chunk(i: int, text: str, retrievable: bool = True) -> Chunk:
    return Chunk(chunk_id=f"doc_x-c-{i:05d}", document_id="doc_x", source_file="f",
                 chunking_strategy=ChunkingStrategy.STRUCTURE, config_id="structure-256",
                 chunk_index=i, content=text, token_count=1, element_type="paragraph",
                 element_ids=["e"], retrievable=retrievable)  # fmt: skip


def test_chunk_embedder_sets_resolvable_refs(tmp_path: Path) -> None:
    fake = FakeOllama()
    embedder = _embedder(fake, tmp_path, document_prefix="d: ")
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta", retrievable=False), _chunk(2, "gamma")]
    stats = ChunkEmbedder(embedder, ["structure-256"]).embed_sets({"structure-256": chunks})
    assert stats["chunks"] == {"structure-256": 2} and stats["space_id"].startswith("dense-ollama")
    assert chunks[1].embedding_ref is None  # parents / non-retrievable chunks are not embedded
    refs = [chunks[0].embedding_ref, chunks[2].embedding_ref]
    assert all(r.startswith(embedder.space.space_id + "/") for r in refs)
    vectors = load_vectors(embedder.cache, refs)
    assert np.allclose(vectors, embedder.embed_documents(["alpha", "gamma"]))
    with pytest.raises(EmbeddingError, match="2 embedding spaces"):
        load_vectors(embedder.cache, [refs[0], "dense-other-x-000000000000/abc"])


def test_validate_dense_and_sparse_dot() -> None:
    with pytest.raises(EmbeddingError, match="NaN"):
        validate_dense(np.array([[np.nan, 1.0]]), count=1, dim=2, source="t")
    with pytest.raises(EmbeddingError, match="all-zero"):
        validate_dense(np.zeros((1, 2)), count=1, dim=2, source="t")
    assert sparse_dot({1: 0.5, 2: 0.2}, {2: 0.5, 3: 1.0}) == pytest.approx(0.1)
