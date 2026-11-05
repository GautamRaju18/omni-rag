"""Dense embeddings through Ollama's ``/api/embed``, with validation, retries and a cache.

Measured 2026-11-05 with Ollama 0.40.14 and ``bge-m3`` (F16 GGUF, 0.69 GB VRAM):
* only dense vectors come back (``embeddings``): no sparse or multi-vector output;
* **the usable input limit is 2,048 tokens incl. the 2 special tokens**, even though the
  model advertises 8,192 and ``/api/ps`` shows an 8,192 context; ``num_ctx`` doesn't change
  it. With the default ``truncate: true``, a ~9,000-token text returned HTTP 200 with only
  2,048 tokens embedded. We always send ``truncate: false`` and check lengths first;
* an empty string still returns a vector (meaningless), so empty input is rejected here;
* throughput: 17 texts/s one at a time, ~43 texts/s in batches of 8–64. A text alone and
  inside a batch gives bit-identical vectors, so caching per text is safe.

Retries: connection errors, timeouts, HTTP 429 and 5xx are retried with exponential backoff.
HTTP 400/413 means a bad input. For a batch we bisect to report *which* input failed.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path

import httpx
import numpy as np

from omnirag.embeddings.base import (
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingSpace,
    validate_dense,
)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_SPECIAL_TOKENS = 2  # <s> ... </s> that Ollama adds around every input


class EmbeddingCache:
    """SQLite file: (space_id, text_sha256) -> float32 vector. Safe to share across threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(model TEXT, sha256 TEXT, dim INTEGER, vector BLOB, PRIMARY KEY (model, sha256))"
        )
        self._lock = threading.Lock()

    def get_many(self, space_id: str, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):  # SQLite parameter limit
                batch = list(keys[i : i + 500])
                marks = ",".join("?" * len(batch))
                sql = (
                    f"SELECT sha256, vector FROM embeddings WHERE model = ? AND sha256 IN ({marks})"  # noqa: S608, E501 (only "?" placeholders)
                )
                rows = self._db.execute(sql, [space_id, *batch]).fetchall()
                out.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return out

    def spaces(self) -> list[tuple[str, int, int]]:
        """``(space_id, dim, count)`` for every space in the cache."""
        with self._lock:
            return self._db.execute(
                "SELECT model, dim, COUNT(*) FROM embeddings GROUP BY model, dim ORDER BY model"
            ).fetchall()

    def put_many(self, space_id: str, items: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                [(space_id, k, len(v), v.astype(np.float32).tobytes()) for k, v in items.items()],
            )
            self._db.commit()


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class OllamaDenseEmbedder:
    """``DenseEmbedder`` backed by Ollama. The space (digest, dim) is resolved on first use."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        expected_dim: int | None = None,
        max_tokens: int = 2048,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        timeout_s: float = 300.0,
        retries: int = 3,
        backoff_s: float = 0.5,
        cache: EmbeddingCache | None = None,
        count_tokens: Callable[[str], int] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model if ":" in model else f"{model}:latest"
        self.expected_dim = expected_dim
        self.max_tokens = max_tokens
        self.query_prefix, self.document_prefix = query_prefix, document_prefix
        self.batch_size, self.retries, self.backoff_s = batch_size, retries, backoff_s
        self.cache = cache
        self.count_tokens = count_tokens
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self.stats = {"embedded": 0, "cached": 0, "seconds": 0.0, "retries": 0}

    # ------------------------------------------------------------------ space
    @cached_property
    def space(self) -> EmbeddingSpace:
        tags = self._call("GET", "/api/tags").json().get("models", [])
        entry = next((m for m in tags if m["name"] == self.model), None)
        if entry is None:
            raise EmbeddingError(f"Model {self.model!r} not found in Ollama (ollama pull it)")
        info = self._call("POST", "/api/show", {"model": self.model}).json()
        arch = info.get("model_info", {}).get("general.architecture", "")
        dim = info.get("model_info", {}).get(f"{arch}.embedding_length")
        if not dim:
            raise EmbeddingError(
                f"{self.model} reports no embedding_length (not an embedding model?)"
            )
        if self.expected_dim is not None and dim != self.expected_dim:
            raise EmbeddingError(f"{self.model} has dimension {dim}, expected {self.expected_dim}")
        return EmbeddingSpace(
            provider="ollama", model=self.model.removesuffix(":latest"), revision=entry["digest"],
            dim=dim, max_tokens=self.max_tokens, query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
        )  # fmt: skip

    # ------------------------------------------------------------------- API
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self.document_prefix + t for t in texts], raw=texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text], raw=[text])[0]

    embed = embed_documents  # the chunking Embedder protocol (semantic chunking)

    def _embed(self, texts: list[str], raw: Sequence[str]) -> np.ndarray:
        space = self.space
        for i, t in enumerate(raw):
            if not t.strip():
                raise EmbeddingInputError(f"Input {i} is empty", index=i)
            if self.count_tokens is not None:
                n = self.count_tokens(texts[i]) + _SPECIAL_TOKENS
                if n > space.max_tokens:
                    raise EmbeddingInputError(
                        f"Input {i} has {n} tokens; {space.model} accepts {space.max_tokens}",
                        index=i,
                    )
        keys = [text_key(t) for t in texts]
        found = self.cache.get_many(space.space_id, list(set(keys))) if self.cache else {}
        missing = list(dict.fromkeys(k for k in keys if k not in found))
        by_key = dict(zip(keys, texts, strict=True))
        started = time.perf_counter()
        new: dict[str, np.ndarray] = {}
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]
            try:
                vectors = self._embed_batch([by_key[k] for k in batch])
            except EmbeddingInputError as exc:
                if exc.text is not None:  # map the rejected text back to the caller's index
                    exc.index = texts.index(exc.text)
                    exc.args = (f"Input {exc.index}: {exc.args[0]}",)
                raise
            new.update(zip(batch, vectors, strict=True))
        if new and self.cache:
            self.cache.put_many(space.space_id, new)
        self.stats["seconds"] += time.perf_counter() - started
        self.stats["embedded"] += len(new)
        self.stats["cached"] += len(keys) - len(new)
        found.update(new)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        try:
            response = self._call(
                "POST", "/api/embed", {"model": self.model, "input": texts, "truncate": False}
            )
        except EmbeddingInputError as exc:
            if len(texts) == 1:
                raise EmbeddingInputError(str(exc), text=texts[0]) from exc
            mid = len(texts) // 2  # bisect: find the input the server rejects
            return np.concatenate([self._embed_batch(texts[:mid]), self._embed_batch(texts[mid:])])
        vectors = np.asarray(response.json().get("embeddings", []), dtype=np.float32)
        return validate_dense(vectors, count=len(texts), dim=self.space.dim, source=self.model)

    def _call(self, method: str, path: str, body: dict | None = None) -> httpx.Response:
        for attempt in range(self.retries + 1):
            try:
                response = self._client.request(method, path, json=body)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt == self.retries:
                    raise EmbeddingError(f"Ollama unreachable ({path}): {exc}") from exc
            else:
                if response.status_code < 400:
                    return response
                detail = _error_text(response)
                if response.status_code == 404:
                    raise EmbeddingError(f"Ollama: {detail} (ollama pull {self.model})")
                if response.status_code in (400, 413):
                    raise EmbeddingInputError(f"Ollama rejected the input: {detail}")
                if response.status_code not in _RETRY_STATUS or attempt == self.retries:
                    raise EmbeddingError(f"Ollama HTTP {response.status_code}: {detail}")
            self.stats["retries"] += 1
            time.sleep(self.backoff_s * 2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text[:200]
