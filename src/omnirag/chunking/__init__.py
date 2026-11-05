"""Chunking engine: turns a document's elements into retrievable chunks (Phase 3)."""

from __future__ import annotations

from functools import partial

from omnirag.chunking.models import Chunk, ChunkingConfig, ChunkingStrategy
from omnirag.chunking.service import Chunker, default_configs
from omnirag.chunking.tokenizer import HFTokenizer, TokenCounter, ensure_tokenizer
from omnirag.core.config import Settings

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


def build_chunker(settings: Settings) -> Chunker:
    configs = default_configs(
        [ChunkingStrategy(s) for s in settings.chunking_strategies],
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        child_size=settings.chunk_child_size,
        parent_size=settings.chunk_parent_size,
        contextual_headers=settings.chunk_contextual_headers,
    )
    return Chunker(partial(load_tokenizer, settings), configs)
