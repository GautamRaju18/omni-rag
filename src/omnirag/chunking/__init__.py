"""Chunking engine: turns a document's elements into retrievable chunks (Phase 3)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from omnirag.chunking.models import Chunk, ChunkingConfig, ChunkingStrategy
from omnirag.chunking.service import Chunker, default_configs
from omnirag.chunking.tokenizer import HFTokenizer, TokenCounter, ensure_tokenizer
from omnirag.core.config import Settings
from omnirag.embeddings import EmbeddingCache, OllamaDenseEmbedder

__all__ = ["Chunk", "Chunker", "ChunkingConfig", "ChunkingStrategy", "build_chunker"]


def load_tokenizer(settings: Settings) -> TokenCounter:
    path = ensure_tokenizer(
        settings.data_dir / "models",
        settings.chunk_tokenizer_repo,
        settings.chunk_tokenizer_revision,
        settings.chunk_tokenizer_sha256,
    )
    name = f"{settings.chunk_tokenizer_repo}@{settings.chunk_tokenizer_revision[:12]}"
    return HFTokenizer(path, name)


def build_embedder(
    settings: Settings, count_tokens: Callable[[str], int] | None = None
) -> OllamaDenseEmbedder:
    """The configured dense model via Ollama, with the shared on-disk cache."""
    return OllamaDenseEmbedder(
        settings.ollama_base_url,
        settings.embedding_model,
        expected_dim=settings.embedding_dim,
        max_tokens=settings.embedding_max_tokens,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
        batch_size=settings.embedding_batch_size,
        cache=EmbeddingCache(settings.data_dir / "cache" / "embeddings.sqlite"),
        count_tokens=count_tokens,
    )


def build_chunker(settings: Settings) -> Chunker:
    strategies = [ChunkingStrategy(s) for s in settings.chunking_strategies]
    configs = default_configs(
        strategies,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        child_size=settings.chunk_child_size,
        parent_size=settings.chunk_parent_size,
        contextual_headers=settings.chunk_contextual_headers,
    )
    chunker = Chunker(partial(load_tokenizer, settings), configs)
    if ChunkingStrategy.SEMANTIC in strategies:
        chunker.embedder = build_embedder(settings, lambda t: chunker.counter.count(t))
    return chunker
