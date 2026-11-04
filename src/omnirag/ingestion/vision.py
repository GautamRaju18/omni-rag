"""Image descriptions from a local vision-language model (VLM) served by Ollama.

Charts, diagrams and photos carry information that OCR can't express ("retention drops below
80% only for AUR-C103"). A VLM can describe it, but **its output is generated text, not a
quote**. Every description is stored as a separate DESCRIPTION element with
``origin=GENERATED``, a fixed policy confidence of 0.5 (unverified) and a link to its IMAGE.
Answers can use it; citations must point at the image, never quote the description as the
document's words.

Runs as an *enrichment* step after the loader. Its elements are appended **after** all loader
elements, so loader element IDs stay identical whether vision is on, off or failing.

VRAM: the VLM (2.1 GB for the default qwen3-vl:2b) doesn't fit next to the chat and embedding
models on a 6 GB GPU. It stays loaded for an ingestion *batch* and is unloaded by
``ImageDescriptionEnricher.close()`` (``IngestionPipeline.close()``), or by Ollama after the
``keep_alive`` timeout, so the chat model can come back.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import ContentOrigin, ElementType, ExtractionMethod

PROMPT_VERSION = "describe-v1"
# Changing this text changes model output: bump PROMPT_VERSION and re-run the benchmark.
PROMPT = (
    "Describe this image for a search index. Be factual and specific.\n"
    "- First line: what kind of image it is (bar chart, line chart, photo, diagram, table, "
    "screenshot...).\n"
    "- Transcribe all visible text, labels and numbers exactly as written.\n"
    "- For charts: name the axes, the series and the values you can read, and the main trend.\n"
    "- If something is unreadable, say so. Never guess values that are not visible.\n"
    "Answer in plain text, at most 150 words."
)
GENERATED_CONFIDENCE = 0.5  # policy value: model output is unverified, not a measurement
_MAX_SIDE = 1280  # px; larger images cost more vision tokens without helping a 4B model


class VisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Description:
    text: str
    model: str
    duration_ms: float
    load_ms: float
    output_tokens: int


def prepare_image(data: bytes) -> bytes:
    """Upright (EXIF), RGB, longest side <= 1280 px, JPEG. Same input -> same bytes."""
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((_MAX_SIDE, _MAX_SIDE))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=90)
    return out.getvalue()


class OllamaVision:
    """Synchronous client (loaders and enrichers run in worker threads, not the event loop)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 180.0,
        keep_alive: str = "2m",
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.keep_alive = keep_alive
        self._client = http_client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def describe(self, image: bytes, *, hint: str | None = None) -> Description:
        prompt = PROMPT
        if hint:
            # Context helps the model name things, but it is told the hint is not the image.
            prompt += f"\nContext from the surrounding document (may be incomplete): {hint}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(prepare_image(image)).decode()],
                }
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "seed": 42, "num_predict": 400},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisionError(f"Ollama vision call failed: {exc}") from exc
        body = response.json()
        return Description(
            text=body["message"]["content"].strip(),
            model=self.model,
            duration_ms=round(body.get("total_duration", 0) / 1e6, 1),
            load_ms=round(body.get("load_duration", 0) / 1e6, 1),
            output_tokens=body.get("eval_count", 0),
        )

    def unload(self) -> None:
        """Free VRAM now instead of waiting for keep_alive to expire."""
        try:
            self._client.post("/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass  # best effort: Ollama unloads on its own after keep_alive


class ImageDescriptionEnricher:
    """Adds a DESCRIPTION child to every IMAGE that has a stored asset.

    Skips images whose OCR text already captures them (scanned text pages): describing a page of
    typed text would only paraphrase the OCR, at several seconds per image.
    """

    name = "vision"

    def __init__(
        self, vision: OllamaVision, *, max_images: int = 20, skip_if_ocr_words: int = 30
    ) -> None:
        self.vision = vision
        self.max_images = max_images
        self.skip_if_ocr_words = skip_if_ocr_words
        # Process-level (not per-file) state: did any document load the model since close()?
        self._model_loaded = False

    def close(self) -> None:
        """Unload the VLM after a batch. Not after every document: measured 2026-11-04, each
        described image cost 6-7 s when every document paid a ~5 s cold load. If close() is
        never reached, the keep_alive timeout unloads the model anyway."""
        if self._model_loaded:
            self.vision.unload()
            self._model_loaded = False

    def enrich(self, ctx: ExtractionContext, read_asset: Callable[[str], Path]) -> dict[str, Any]:
        images = [e for e in ctx.elements if e.element_type is ElementType.IMAGE and e.asset_ref]
        children: dict[str, list] = {}
        for e in ctx.elements:
            if e.parent_element_id:
                children.setdefault(e.parent_element_id, []).append(e)
        stats = {"model": self.vision.model, "prompt": PROMPT_VERSION, "described": 0,
                 "skipped_text_images": 0, "total_ms": 0.0}  # fmt: skip
        for image in images:
            kids = children.get(image.element_id, [])
            ocr_words = sum(
                k.metadata.get("words", 0) for k in kids if k.origin is ContentOrigin.OCR
            )
            if ocr_words >= self.skip_if_ocr_words:
                stats["skipped_text_images"] += 1
                continue
            if stats["described"] >= self.max_images:
                ctx.warn(f"Only the first {self.max_images} images were described")
                break
            hint = " ".join(
                t for t in [image.content, *(k.content for k in kids
                            if k.element_type is ElementType.CAPTION)] if t
            )  # fmt: skip
            try:
                self._model_loaded = True  # a failed call may still have loaded it
                result = self.vision.describe(
                    read_asset(image.asset_ref).read_bytes(), hint=hint or None
                )
            except VisionError as exc:
                ctx.warn(f"Image description skipped: {exc}")
                break  # Ollama down or model missing: don't retry for every image
            ctx.add(
                ElementType.DESCRIPTION,
                result.text,
                extraction_method=ExtractionMethod.VLM,
                origin=ContentOrigin.GENERATED,
                confidence=GENERATED_CONFIDENCE,
                page_number=image.page_number,
                section_path=image.section_path,
                parent_id=image.element_id,
                metadata={
                    "model": result.model,
                    "prompt": PROMPT_VERSION,
                    "hint_used": bool(hint),
                    "duration_ms": result.duration_ms,
                    "load_ms": result.load_ms,
                    "output_tokens": result.output_tokens,
                },
            )
            stats["described"] += 1
            stats["total_ms"] += result.duration_ms
        return stats
