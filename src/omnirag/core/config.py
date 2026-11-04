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

    # --- OCR (Tesseract CLI, installed with `sudo apt install tesseract-ocr`)
    ocr_enabled: bool = True
    ocr_tesseract_cmd: str = "tesseract"
    # Tesseract language codes joined with "+", e.g. "eng+deu" (needs tesseract-ocr-deu).
    ocr_languages: str = Field(default="eng", pattern=r"^[A-Za-z_]+(\+[A-Za-z_]+)*$")
    ocr_dpi: int = Field(default=300, ge=72, le=600)  # render resolution for scanned pages
    ocr_timeout_s: float = Field(default=120.0, gt=0)  # per page / image
    ocr_max_pages: int = Field(default=50, ge=0)  # per document; OCR costs seconds per page

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def required_ollama_models(self) -> list[str]:
        """Models that must be pulled for the platform to be fully functional."""
        return [self.generation_model, self.embedding_model]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached; tests build their own ``Settings``)."""
    return Settings()
