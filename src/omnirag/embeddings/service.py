"""Embed chunk sets and give every indexed chunk a versioned, content-addressed reference.

``chunk.embedding_ref = "<space_id>/<sha256 of the embedded text>"``:
* **versioned**: the space ID includes the model's exact revision (Ollama digest);
* **content-addressed**: the same text in the same space always has the same ref, so a vector
  is never computed twice, and a chunk whose text changed gets a new ref;
* **resolvable**: Phase 5 loads the vector from the embedding cache by this ref and stores it
  on the Chunk node in Neo4j.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from omnirag.chunking.models import Chunk
from omnirag.embeddings.base import DenseEmbedder, EmbeddingError
from omnirag.embeddings.ollama import EmbeddingCache, text_key


def make_ref(space_id: str, embedded_text: str) -> str:
    return f"{space_id}/{text_key(embedded_text)}"


class ChunkEmbedder:
    def __init__(self, embedder: DenseEmbedder, config_ids: Sequence[str]) -> None:
        self.embedder = embedder
        self.config_ids = list(config_ids)

    def embed_sets(self, sets: dict[str, list[Chunk]]) -> dict[str, Any]:
        """Embed the configured chunk sets in place (sets ``embedding_ref``); return stats."""
        space = self.embedder.space
        started = time.perf_counter()
        counts: dict[str, int] = {}
        for config_id in self.config_ids:
            chunks = [c for c in sets.get(config_id, []) if c.retrievable]
            if not chunks:
                continue
            self.embedder.embed_documents([c.embedding_text() for c in chunks])
            for c in chunks:
                c.embedding_ref = make_ref(
                    space.space_id, space.document_prefix + c.embedding_text()
                )
            counts[config_id] = len(chunks)
        return {
            "space_id": space.space_id,
            "dim": space.dim,
            "chunks": counts,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }


def load_vectors(cache: EmbeddingCache, refs: Sequence[str]) -> np.ndarray:
    """Vectors for ``embedding_ref``s (all from one space), in order. Missing refs raise."""
    spaces = {r.split("/", 1)[0] for r in refs}
    if len(spaces) != 1:
        raise EmbeddingError(f"Refs span {len(spaces)} embedding spaces; vectors are incomparable")
    space_id = spaces.pop()
    keys = [r.split("/", 1)[1] for r in refs]
    found = cache.get_many(space_id, keys)
    missing = [r for r, k in zip(refs, keys, strict=True) if k not in found]
    if missing:
        raise EmbeddingError(f"{len(missing)} vectors missing from the cache, e.g. {missing[0]}")
    return np.stack([found[k] for k in keys])
