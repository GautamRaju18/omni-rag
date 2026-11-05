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
    chunk_size: int = Field(default=512, ge=16, le=8192)
    chunk_overlap: int = Field(default=64, ge=0)
    chunk_child_size: int = Field(default=128, ge=16)
    chunk_parent_size: int = Field(default=2048, ge=16)
    chunk_contextual_headers: bool = False
    # The embedding model's tokenizer, pinned: revision + SHA-256 of tokenizer.json.
    chunk_tokenizer_repo: str = "BAAI/bge-m3"
    chunk_tokenizer_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    chunk_tokenizer_sha256: str = "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"

    # --- PDF engine (ADR-0005): "auto" routes multi-column documents to Docling's layout model
    # and everything else to PyMuPDF (~40x faster); "pymupdf" / "docling" force one engine.
    # Scanned pages always use PyMuPDF + Tesseract.
    pdf_engine: Literal["auto", "pymupdf", "docling"] = "auto"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

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
