"""Vision enrichment with a fake Ollama (httpx.MockTransport): no GPU or model needed."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import (
    ContentOrigin,
    ElementType,
    ExtractionMethod,
    IngestionStatus,
)
from omnirag.ingestion.vision import (
    GENERATED_CONFIDENCE,
    ImageDescriptionEnricher,
    OllamaVision,
    VisionError,
    prepare_image,
)


def _png(size=(64, 32)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, "red").save(out, format="PNG")
    return out.getvalue()


class FakeOllama:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/generate":  # unload
            return httpx.Response(200, json={"done": True})
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "model not found"})
        return httpx.Response(
            200,
            json={
                "message": {"content": " Bar chart of retention per cell. "},
                "total_duration": 2_500_000_000,
                "load_duration": 900_000_000,
                "eval_count": 42,
            },
        )


def _vision(fake: FakeOllama) -> OllamaVision:
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(fake))
    return OllamaVision("http://ollama", "vlm:test", http_client=client)


def test_prepare_image_downscales_and_applies_exif() -> None:
    img = Image.new("RGB", (4000, 1000), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # display rotated 90° clockwise
    out = io.BytesIO()
    img.save(out, format="JPEG", exif=exif.tobytes())
    with Image.open(io.BytesIO(prepare_image(out.getvalue()))) as prepared:
        assert prepared.size == (320, 1280)  # upright (portrait) and longest side 1280


def test_describe_sends_deterministic_request() -> None:
    fake = FakeOllama()
    result = _vision(fake).describe(_png(), hint="Figure 1: retention")
    path, body = fake.requests[0]
    assert path == "/api/chat" and body["stream"] is False
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == 42
    assert "Figure 1: retention" in body["messages"][0]["content"]
    assert base64.b64decode(body["messages"][0]["images"][0])[:3] == b"\xff\xd8\xff"  # JPEG
    assert result.text == "Bar chart of retention per cell."
    assert (result.duration_ms, result.load_ms, result.output_tokens) == (2500.0, 900.0, 42)


def test_describe_error_raises_vision_error() -> None:
    with pytest.raises(VisionError):
        _vision(FakeOllama(status=404)).describe(_png())


def _ctx_with_image(tmp_path: Path) -> tuple[ExtractionContext, dict[str, Path]]:
    files: dict[str, Path] = {}

    def saver(name: str, data: bytes) -> str:
        files[name] = tmp_path / name
        files[name].write_bytes(data)
        return name

    ctx = ExtractionContext("doc_v", "r.pdf", asset_saver=saver)
    m = ExtractionMethod.PYMUPDF
    ctx.add(ElementType.PARAGRAPH, "Intro.", extraction_method=m)
    img = ctx.add(ElementType.IMAGE, "", extraction_method=m, page_number=2,
                  section_path=["Results"], asset_ref=ctx.save_asset("a.png", _png()))  # fmt: skip
    ctx.add(ElementType.CAPTION, "Figure 1: retention", extraction_method=m, parent_id=img)
    return ctx, files


def test_enricher_appends_generated_description(tmp_path: Path) -> None:
    ctx, files = _ctx_with_image(tmp_path)
    before = [e.element_id for e in ctx.elements]
    fake = FakeOllama()
    enricher = ImageDescriptionEnricher(_vision(fake))
    stats = enricher.enrich(ctx, lambda ref: files[ref])
    assert [e.element_id for e in ctx.elements[:3]] == before  # loader IDs untouched
    desc = ctx.elements[-1]
    assert desc.element_type is ElementType.DESCRIPTION
    assert desc.origin is ContentOrigin.GENERATED and desc.confidence == GENERATED_CONFIDENCE
    assert desc.parent_element_id == ctx.elements[1].element_id
    assert (desc.page_number, desc.section_path) == (2, ["Results"])
    assert desc.metadata["model"] == "vlm:test" and desc.metadata["hint_used"] is True
    assert stats["described"] == 1
    assert fake.requests[-1][0] == "/api/chat"  # stays loaded for the rest of the batch
    enricher.close()
    enricher.close()  # second close: nothing loaded, no request
    assert [p for p, _ in fake.requests] == ["/api/chat", "/api/generate"]


def test_enricher_skips_text_images_already_ocred(tmp_path: Path) -> None:
    ctx, files = _ctx_with_image(tmp_path)
    ctx.add(ElementType.PARAGRAPH, "lots of words", extraction_method=ExtractionMethod.TESSERACT,
            origin=ContentOrigin.OCR, parent_id=ctx.elements[1].element_id,
            metadata={"words": 45})  # fmt: skip
    fake = FakeOllama()
    stats = ImageDescriptionEnricher(_vision(fake)).enrich(ctx, lambda ref: files[ref])
    assert stats["skipped_text_images"] == 1 and stats["described"] == 0
    assert fake.requests == []  # no call, no unload


def test_enricher_failure_is_a_warning(tmp_path: Path) -> None:
    ctx, files = _ctx_with_image(tmp_path)
    stats = ImageDescriptionEnricher(_vision(FakeOllama(status=404))).enrich(
        ctx, lambda ref: files[ref]
    )
    assert stats["described"] == 0 and "description skipped" in ctx.warnings[0]
    assert all(e.element_type is not ElementType.DESCRIPTION for e in ctx.elements)


def test_pipeline_survives_crashing_enricher(tmp_path: Path) -> None:
    from omnirag.ingestion.loaders import default_loaders
    from omnirag.ingestion.pipeline import IngestionPipeline
    from omnirag.ingestion.store import DocumentStore

    class Boom:
        name = "boom"

        def enrich(self, ctx, read_asset):
            raise RuntimeError("kaput")

    path = tmp_path / "n.txt"
    path.write_text("hello")
    store = DocumentStore(tmp_path / "data")
    pipeline = IngestionPipeline(store, default_loaders(), max_bytes=1024, enrichers=[Boom()])
    record = pipeline.ingest(path).record
    assert record.status is IngestionStatus.COMPLETED and record.element_count == 1
    assert "Enricher 'boom' failed: RuntimeError: kaput" in record.warnings
