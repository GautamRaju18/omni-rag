"""Multimodal document ingestion: validation, loaders, normalization, enrichment, storage."""

from __future__ import annotations

from omnirag.core.config import Settings
from omnirag.ingestion.loaders import default_loaders
from omnirag.ingestion.pipeline import Enricher, IngestionPipeline
from omnirag.ingestion.store import DocumentStore
from omnirag.ingestion.vision import ImageDescriptionEnricher, OllamaVision


def build_enrichers(settings: Settings) -> list[Enricher]:
    enrichers: list[Enricher] = []
    if settings.vision_enabled:
        vision = OllamaVision(
            settings.ollama_base_url, settings.vision_model, timeout_s=settings.vision_timeout_s
        )
        enrichers.append(ImageDescriptionEnricher(vision, max_images=settings.vision_max_images))
    return enrichers


def build_pipeline(settings: Settings) -> IngestionPipeline:
    """Wire the default pipeline from settings (used by the CLI and, in 2e, the API)."""
    return IngestionPipeline(
        DocumentStore(settings.data_dir),
        default_loaders(settings),
        max_bytes=settings.max_upload_bytes,
        enrichers=build_enrichers(settings),
    )
