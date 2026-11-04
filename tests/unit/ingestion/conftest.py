from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.loaders.base import Loader

LoaderRunner = Callable[[Loader, str, str | bytes], tuple[ExtractionContext, dict]]


@pytest.fixture
def run_loader(tmp_path: Path) -> LoaderRunner:
    """Write ``content`` to ``tmp_path/name`` and run ``loader`` on it."""

    def _run(loader: Loader, name: str, content: str | bytes) -> tuple[ExtractionContext, dict]:
        path = tmp_path / name
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)
        ctx = ExtractionContext("doc_test", name)
        metadata = loader.load(path, ctx)
        return ctx, metadata

    return _run
