"""bge-m3 embeddings through Ollama's ``/api/embed``, with a persistent cache.

Measured 2026-11-05 (bge-m3 on the RTX 3060, 0.69 GB VRAM): 17 texts/s one at a time, ~43
texts/s in batches of 8–64. A text embedded alone and inside a batch gave bit-identical
vectors, so caching per text is safe. The cache is keyed by (model, SHA-256 of the text):
the same chunk text produced by several strategies, or a re-run, is embedded once.

Vectors are L2-normalised (bge-m3 returns norm 1.0; we re-normalise defensively), so a dot
product is the cosine similarity.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np


class EmbeddingError(RuntimeError):
    pass


class EmbeddingCache:
    """SQLite file: (model, text_sha256) -> float32 vector. Safe to share across threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(model TEXT, sha256 TEXT, dim INTEGER, vector BLOB, PRIMARY KEY (model, sha256))"
        )
        self._lock = threading.Lock()

    def get_many(self, model: str, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(keys), 500):  # SQLite parameter limit
                batch = list(keys[i : i + 500])
                marks = ",".join("?" * len(batch))
                sql = (
                    f"SELECT sha256, vector FROM embeddings WHERE model = ? AND sha256 IN ({marks})"  # noqa: S608, E501 (only "?" placeholders)
                )
                rows = self._db.execute(sql, [model, *batch]).fetchall()
                out.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return out

    def put_many(self, model: str, items: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                [(model, k, len(v), v.astype(np.float32).tobytes()) for k, v in items.items()],
            )
            self._db.commit()


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        batch_size: int = 32,
        timeout_s: float = 300.0,
        cache: EmbeddingCache | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.cache = cache
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self.stats = {"embedded": 0, "cached": 0, "seconds": 0.0}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """``(len(texts), dim)`` float32, L2-normalised, in input order."""
        keys = [_key(t) for t in texts]
        found = self.cache.get_many(self.model, list(set(keys))) if self.cache else {}
        missing = list(dict.fromkeys(k for k in keys if k not in found))
        by_key = {k: t for k, t in zip(keys, texts, strict=True)}
        started = time.perf_counter()
        new: dict[str, np.ndarray] = {}
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i : i + self.batch_size]
            vectors = self._request([by_key[k] for k in batch])
            new.update(zip(batch, vectors, strict=True))
        if new and self.cache:
            self.cache.put_many(self.model, new)
        self.stats["seconds"] += time.perf_counter() - started
        self.stats["embedded"] += len(new)
        self.stats["cached"] += len(keys) - len(new)
        found.update(new)
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)

    def _request(self, texts: list[str]) -> list[np.ndarray]:
        try:
            response = self._client.post("/api/embed", json={"model": self.model, "input": texts})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embed call failed: {exc}") from exc
        vectors = np.asarray(response.json()["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return list(vectors / np.where(norms == 0, 1, norms))
