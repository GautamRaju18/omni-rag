"""Multimodal document ingestion: validation, loaders, normalization, storage."""

from __future__ import annotations

from omnirag.core.config import Settings
from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.ingestion.store import DocumentStore


def build_pipeline(settings: Settings) -> IngestionPipeline:
    """Wire the default pipeline from settings (used by the CLI and, in 2e, the API)."""
    return IngestionPipeline(
        DocumentStore(settings.data_dir),
        default_loaders(settings),
        max_bytes=settings.max_upload_bytes,
    )
