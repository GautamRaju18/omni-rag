"""Loader registry: one loader per ``FileKind``."""

from __future__ import annotations

from omnirag.ingestion.loaders.base import Loader
from omnirag.ingestion.loaders.csv_loader import CsvLoader
from omnirag.ingestion.loaders.html_loader import HtmlLoader
from omnirag.ingestion.loaders.json_loader import JsonLoader
from omnirag.ingestion.loaders.markdown_loader import MarkdownLoader
from omnirag.ingestion.loaders.text_loader import TextLoader
from omnirag.ingestion.models import FileKind


def default_loaders() -> dict[FileKind, Loader]:
    """All loaders available in this build. Office, PDF and image loaders arrive in 2b–2d."""
    loaders: list[Loader] = [
        TextLoader(),
        MarkdownLoader(),
        HtmlLoader(),
        JsonLoader(),
        CsvLoader(),
    ]
    return {kind: loader for loader in loaders for kind in loader.kinds}
