"""Application configuration.

All runtime configuration comes from environment variables (prefix ``OMNIRAG_``) or a local
``.env`` file. Nothing secret is hardcoded; the Neo4j password is a ``SecretStr`` so it never
appears in logs, reprs, or tracebacks.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated settings for the whole application."""

    model_config = SettingsConfigDict(
        env_prefix="OMNIRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds non-prefixed vars for docker compose
    )

    # --- app
    env: Literal["dev", "test", "prod"] = "dev"
    app_name: str = "OmniRAG"
    api_prefix: str = "/api/v1"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # --- neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    neo4j_database: str = "neo4j"
    neo4j_connection_timeout_s: float = Field(default=5.0, gt=0)
    neo4j_health_timeout_s: float = Field(default=5.0, gt=0)

    # --- ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_s: float = Field(default=120.0, gt=0)
    ollama_health_timeout_s: float = Field(default=3.0, gt=0)
    ollama_connect_retries: int = Field(default=2, ge=0, le=5)
    generation_model: str = "qwen3:4b-instruct"
    embedding_model: str = "bge-m3"

    # --- ingestion
    # Originals, extracted elements and assets. Keep this on the WSL ext4 disk: fast I/O and
    # plenty of space (the repo drive P: has only a few GB free).
    data_dir: Path = Field(default_factory=lambda: Path.home() / "omnirag-data")
    max_upload_mb: int = Field(default=100, gt=0)
    # Background ingestion (API). One worker: OCR, Docling and the vision model all compete
    # for the same CPU/GPU, so parallel jobs would mostly slow each other down.
    ingestion_workers: int = Field(default=1, ge=1, le=4)
    ingestion_queue_max: int = Field(default=100, ge=1, le=10_000)
    ingestion_shutdown_grace_s: float = Field(default=30.0, ge=0)

    # --- OCR (Tesseract CLI, installed with `sudo apt install tesseract-ocr`)
    ocr_enabled: bool = True
    ocr_tesseract_cmd: str = "tesseract"
    # Tesseract language codes joined with "+", e.g. "eng+deu" (needs tesseract-ocr-deu).
    ocr_languages: str = Field(default="eng", pattern=r"^[A-Za-z_]+(\+[A-Za-z_]+)*$")
    ocr_dpi: int = Field(default=300, ge=72, le=600)  # render resolution for scanned pages
    ocr_timeout_s: float = Field(default=120.0, gt=0)  # per page / image
    ocr_max_pages: int = Field(default=50, ge=0)  # per document; OCR costs seconds per page

    # --- image descriptions (vision-language model via Ollama)
    vision_enabled: bool = True
    # Chosen by scripts/benchmark_vision.py (2026-11-04): 8/8 chart label-value pairs, ~1-2 s
    # per image warm, 2.1 GB VRAM. The 4B variant missed every pairing; gemma3:4b was 2x slower.
    vision_model: str = "qwen3-vl:2b-instruct"
    vision_timeout_s: float = Field(default=180.0, gt=0)  # per image, includes cold load
    vision_max_images: int = Field(default=20, ge=0)  # per document

    # --- chunking (Phase 3). Sizes are bge-m3 tokens; every strategy runs at ingestion (they
    # take milliseconds) so they can be compared later on the same documents.
    chunking_enabled: bool = True
    chunking_strategies: list[str] = Field(
        default=[
            "fixed",
            "recursive",
            "structure",
            "table",
            "multimodal",
            "hierarchical",
            "parent_child",
        ]  # fmt: skip
    )
    # Defaults follow the Phase 3b benchmark (multimodal-256 + headers was the best
    # chunk-sized dense configuration on the seed set; revisit after Phase 12).
    chunk_size: int = Field(default=256, ge=16, le=8192)
    chunk_overlap: int = Field(default=32, ge=0)
    chunk_child_size: int = Field(default=128, ge=16)
    # <= 2046: the measured Ollama bge-m3 input limit (2,048 incl. 2 special tokens).
    chunk_parent_size: int = Field(default=2000, ge=16)
    chunk_contextual_headers: bool = True
    # The embedding model's tokenizer, pinned: revision + SHA-256 of tokenizer.json.
    chunk_tokenizer_repo: str = "BAAI/bge-m3"
    chunk_tokenizer_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    chunk_tokenizer_sha256: str = "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"

    # --- embeddings (Phase 4). Which chunk sets get embedded at ingestion, and how.
    embedding_enabled: bool = True
    embedding_index_configs: list[str] = Field(default=["multimodal-256-ctx"])
    embedding_dim: int = Field(default=1024, gt=0)  # validated against the model on first use
    # Largest input incl. special tokens. Measured 2026-11-05 (Ollama 0.40.14, bge-m3): 2,048,
    # not the advertised 8,192; longer inputs are rejected (we never let Ollama truncate).
    embedding_max_tokens: int = Field(default=2048, gt=2)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_query_prefix: str = ""  # bge-m3 needs none; e.g. "search_query: " for nomic
    embedding_document_prefix: str = ""

    # --- Neo4j knowledge graph (Phase 5, ADR-0011). Every ingested document is written to the
    # graph; a graph failure is a warning (reload with `python -m omnirag.knowledge_base load`).
    graph_enabled: bool = True
    # Chunk sets stored as Chunk nodes; empty = the embedded sets (embedding_index_configs).
    graph_chunk_configs: list[str] = Field(default_factory=list)
    graph_vector_index: str = "chunk_embedding"
    # Neo4j 2026.x defaults new vector indexes to BINARY; NONE keeps exact float32 vectors.
    graph_vector_quantization: Literal["NONE", "SCALAR", "BINARY"] = "NONE"
    graph_fulltext_index: str = "chunk_fulltext"
    # "classic" keeps tokens with digits whole (aur-c103, e-7): best BM25 on the seed set
    # (Phase 6 benchmark, ADR-0013). Changing it needs `init --recreate-fulltext-index`.
    graph_fulltext_analyzer: str = "classic"

    # --- entity/relation extraction (Phase 5b, ADR-0012). Off at ingestion by default: about
    # 15 s per chunk on a 6 GB GPU (measured). Run `python -m omnirag.knowledge_base extract`.
    extraction_enabled: bool = False
    extraction_model: str = "qwen3:4b-instruct"
    extraction_config: str = ""  # chunk set to extract from; empty = the first graph set
    extraction_timeout_s: float = Field(default=300.0, gt=0)
    extraction_num_ctx: int = Field(default=8192, ge=2048)

    # --- retrieval (Phase 6, ADR-0013): dense (vector index) and lexical (BM25 full-text).
    retrieval_config: str = ""  # chunk set to search; empty = the first graph set
    retrieval_k: int = Field(default=5, ge=1, le=50)
    # Post-filters (no in-index support) fetch k × overfetch candidates, at most max_fetch.
    retrieval_overfetch: int = Field(default=10, ge=1, le=100)
    retrieval_max_fetch: int = Field(default=200, ge=1, le=2000)
    # How technical identifiers (AUR-C103, E-7) enter the BM25 query: plain terms, an extra
    # boosted exact phrase, or a required phrase.
    retrieval_identifier_mode: Literal["terms", "boost", "require"] = "require"
    # --- graph retrieval + fusion (Phase 7, ADR-0014)
    retrieval_graph_max_hops: int = Field(default=2, ge=0, le=3)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    retrieval_fusion_depth: int = Field(default=20, ge=1, le=200)  # candidates per member
    # Per-retriever RRF weights, e.g. {"graph": 0.5}; missing = 1.0.
    retrieval_rrf_weights: dict[str, float] = Field(default_factory=dict)
    # Mode used when a request names none. dense+bm25 measured best on both eval sets;
    # adding graph lowered MRR (ADR-0014).
    retrieval_default_mode: Literal[
        "dense", "bm25", "graph", "dense+bm25", "dense+bm25+graph",
        "dense+bm25+rerank", "dense+bm25+graph+rerank",
    ] = "dense+bm25"  # fmt: skip

    # --- cross-encoder reranking (Phase 8, ADR-0015). Weights load on the first rerank.
    rerank_enabled: bool = True
    rerank_model: Literal["bge-reranker-v2-m3", "bge-reranker-base"] = "bge-reranker-v2-m3"
    rerank_device: Literal["auto", "cpu", "cuda"] = "auto"  # auto: CUDA if available, else CPU
    rerank_depth: int = Field(default=20, ge=1, le=200)  # first-stage candidates re-scored
    rerank_max_length: int = Field(default=512, ge=32, le=8192)  # tokens per (query, chunk)
    # batch 4 beat 16 on CPU (less padding): 11.8 s vs 17.5 s for 20 pairs (v2-m3, measured)
    rerank_batch_size: int = Field(default=4, ge=1, le=256)
    rerank_threads: int = Field(default=8, ge=1, le=64)

    # --- corrective retrieval workflow (Phase 9a, ADR-0016): defaults for /retrieval/advanced
    workflow_transform: Literal["none", "rewrite", "multi_query", "hyde", "multi_query+hyde"] = (
        "none"
    )
    workflow_multi_query_n: int = Field(default=3, ge=1, le=6)
    workflow_grade: bool = True  # LLM context grader after the identifier check
    workflow_max_retries: int = Field(default=1, ge=0, le=3)
    workflow_rerank: bool = False  # CPU reranking costs 10-17 s (ADR-0015)
    workflow_llm_timeout_s: float = Field(default=60.0, gt=0)

    # --- PDF engine (ADR-0005): "auto" routes multi-column documents to Docling's layout model
    # and everything else to PyMuPDF (~40x faster); "pymupdf" / "docling" force one engine.
    # Scanned pages always use PyMuPDF + Tesseract.
    pdf_engine: Literal["auto", "pymupdf", "docling"] = "auto"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def graph_configs(self) -> list[str]:
        return self.graph_chunk_configs or self.embedding_index_configs

    @property
    def extraction_config_id(self) -> str:
        return self.extraction_config or self.graph_configs[0]

    @property
    def retrieval_config_id(self) -> str:
        return self.retrieval_config or self.graph_configs[0]

    @property
    def required_ollama_models(self) -> list[str]:
        """Models that must be pulled for the platform to be fully functional."""
        models = [self.generation_model, self.embedding_model]
        if self.vision_enabled:
            models.append(self.vision_model)
        return models


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached; tests build their own ``Settings``)."""
    return Settings()
