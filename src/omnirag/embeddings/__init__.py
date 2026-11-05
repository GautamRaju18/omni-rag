"""Embedding engine (Phase 4): model-agnostic interfaces, Ollama dense embeddings, a local
BGE-M3 model for sparse/multi-vector investigation, and chunk-set embedding."""

from omnirag.embeddings.base import (
    DenseEmbedder,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingKind,
    EmbeddingSpace,
    SparseEmbedder,
)
from omnirag.embeddings.ollama import EmbeddingCache, OllamaDenseEmbedder

__all__ = [
    "DenseEmbedder",
    "EmbeddingCache",
    "EmbeddingError",
    "EmbeddingInputError",
    "EmbeddingKind",
    "EmbeddingSpace",
    "OllamaDenseEmbedder",
    "SparseEmbedder",
]
