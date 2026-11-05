"""Model-agnostic embedding interfaces.

Retrieval code depends only on these interfaces and on ``EmbeddingSpace.space_id``. Swapping
the model (another Ollama model, a local Hugging Face model, an API) never touches retrieval.

**Versioning.** An ``EmbeddingSpace`` names exactly what produced a vector: provider, model,
*revision* (an Ollama content digest or a Hugging Face commit), kind, dimension and the
query/document prefixes. Vectors from different spaces are never comparable. ``space_id`` is
part of every cache key and every chunk's ``embedding_ref``, so re-pulling a model with
different weights changes the space and stale vectors can't be mixed in silently.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict


class EmbeddingKind(StrEnum):
    DENSE = "dense"  # one vector per text
    SPARSE = "sparse"  # token id -> weight (learned lexical weights)
    MULTIVECTOR = "multivector"  # one vector per token (ColBERT-style late interaction)


class EmbeddingError(RuntimeError):
    """The embedding backend failed (unreachable, model missing, bad response)."""


class EmbeddingInputError(EmbeddingError, ValueError):
    """One input can't be embedded as given (empty, or longer than the model accepts)."""

    def __init__(self, message: str, *, index: int | None = None, text: str | None = None) -> None:
        super().__init__(message)
        self.index = index  # position in the caller's input list
        self.text = text  # the rejected text, when the backend (not our checks) rejected it


class EmbeddingSpace(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str  # "ollama", "hf"
    model: str  # "bge-m3"
    revision: str  # Ollama digest / Hugging Face commit
    kind: EmbeddingKind = EmbeddingKind.DENSE
    dim: int  # vector size (dense/multivector) or vocabulary size (sparse)
    max_tokens: int  # largest input incl. special tokens the backend accepts (measured)
    normalized: bool = True
    query_prefix: str = ""  # e.g. "search_query: " for nomic-embed-text; "" for bge-m3
    document_prefix: str = ""

    @property
    def space_id(self) -> str:
        model = re.sub(r"[^a-z0-9]+", "-", self.model.lower()).strip("-")
        return f"{self.kind.value}-{self.provider}-{model}-{self.revision[:12]}"


class DenseEmbedder(Protocol):
    @property
    def space(self) -> EmbeddingSpace: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """``(n, dim)`` float32, L2-normalised; the document prefix is added here."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """``(dim,)`` float32, L2-normalised; the query prefix is added here."""
        ...


SparseVector = dict[int, float]


class SparseEmbedder(Protocol):
    @property
    def space(self) -> EmbeddingSpace: ...

    def embed_documents_sparse(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def embed_query_sparse(self, text: str) -> SparseVector: ...


def validate_dense(vectors: np.ndarray, *, count: int, dim: int, source: str) -> np.ndarray:
    """Dimension, finiteness and non-zero checks, then L2 normalisation."""
    if vectors.ndim != 2 or vectors.shape != (count, dim):
        raise EmbeddingError(f"{source}: expected {count}×{dim} vectors, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise EmbeddingError(f"{source}: response contains NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise EmbeddingError(f"{source}: response contains an all-zero vector")
    return (vectors / norms).astype(np.float32)


def sparse_dot(a: SparseVector, b: SparseVector) -> float:
    """BGE-M3 lexical matching score: sum of weight products over shared tokens."""
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return float(sum(w * large[t] for t, w in small.items() if t in large))
