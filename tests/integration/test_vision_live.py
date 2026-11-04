"""Live vision-model check against Ollama (run with: uv run pytest -m integration)."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf
import pytest

from omnirag.core.config import Settings
from omnirag.ingestion.vision import OllamaVision

pytestmark = pytest.mark.integration

SAMPLE = Path(__file__).parents[2] / "data" / "samples" / "aurora_safety_review.pdf"
VALUES = ["91.4", "88.0", "72.5", "83.1", "89.6", "90.2", "94.8", "95.1"]


def _chart_png() -> bytes:
    with pymupdf.open(SAMPLE) as doc:
        xref = doc[1].get_images()[0][0]  # the retention bar chart on page 2
        return doc.extract_image(xref)["image"]


def test_vision_model_reads_the_retention_chart() -> None:
    settings = Settings()
    vision = OllamaVision(settings.ollama_base_url, settings.vision_model, timeout_s=300)
    try:
        result = vision.describe(_chart_png())
    finally:
        vision.unload()
    flat = re.sub(r"\s+", "", result.text)
    read = [v for v in VALUES if v in flat or v.removesuffix(".0") in flat]
    # A small model may drop a value; it must not drop most of them.
    assert len(read) >= 6, result.text
    assert "C103" in result.text
