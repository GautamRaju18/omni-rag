"""Embedding engine. Phase 3b brings the minimal client needed for semantic chunking and the
chunking benchmark; Phase 4 adds storage in Neo4j and index management."""

from omnirag.embeddings.ollama import EmbeddingCache, EmbeddingError, OllamaEmbedder

__all__ = ["EmbeddingCache", "EmbeddingError", "OllamaEmbedder"]
