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


def build_pipeline(settings: Settings, store: DocumentStore | None = None) -> IngestionPipeline:
    """Wire the default pipeline from settings (used by the CLI and the API)."""
    # Local imports: chunking and embeddings import ingestion models.
    from omnirag.chunking import build_chunker, build_embedder
    from omnirag.embeddings.service import ChunkEmbedder

    chunker = build_chunker(settings) if settings.chunking_enabled else None
    embedder = None
    if chunker is not None and settings.embedding_enabled:
        # Token limits are checked with the chunker's tokenizer (loaded once, shared).
        dense = build_embedder(settings, lambda text: chunker.counter.count(text))
        embedder = ChunkEmbedder(dense, settings.embedding_index_configs)
    graph = None
    if chunker is not None and settings.graph_enabled:
        graph = build_graph_indexer(settings)
    return IngestionPipeline(
        store or DocumentStore(settings.data_dir),
        default_loaders(settings),
        max_bytes=settings.max_upload_bytes,
        enrichers=build_enrichers(settings),
        chunker=chunker,
        embedder=embedder,
        graph=graph,
    )


def build_graph_indexer(settings: Settings):  # -> GraphIndexer (lazy import, see above)
    from omnirag.embeddings.ollama import EmbeddingCache
    from omnirag.knowledge_base.graph_builder import GraphIndexer
    from omnirag.knowledge_base.graph_store import GraphStore

    store = GraphStore(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
        connection_timeout_s=settings.neo4j_connection_timeout_s,
    )
    return GraphIndexer(
        store,
        EmbeddingCache(settings.data_dir / "cache" / "embeddings.sqlite"),
        settings.graph_configs,
        vector_dim=settings.embedding_dim,
        schema_options={
            "vector_index": settings.graph_vector_index,
            "quantization": settings.graph_vector_quantization,
            "fulltext_index": settings.graph_fulltext_index,
            "analyzer": settings.graph_fulltext_analyzer,
        },
    )
