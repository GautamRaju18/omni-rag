"""Images (.png / .jpg / .jpeg): keep the original, OCR any text in it.

* The uploaded file *is* the asset. The IMAGE element points at the stored original
  (``ctx.original_ref``) instead of saving a second copy.
* **EXIF orientation.** Phone cameras store pixels sideways plus a tag that says "rotate 90°
  when displaying". Tesseract ignores the tag, so OCR runs on ``ImageOps.exif_transpose(img)``,
  the image as a person sees it. OCR boxes refer to that upright image.
* **Decompression bombs.** A tiny PNG can declare 50,000 x 50,000 pixels. Pillow refuses images
  above ``2 x Image.MAX_IMAGE_PIXELS`` (~179 MP) when opening, before decoding anything.
* OCR fragments below 30% confidence are dropped. On photos without text, Tesseract "reads"
  texture as ``|||`` or ``~=``. The IMAGE element stays either way; Phase 2d adds a vision-model
  description with ``origin=GENERATED``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import (
    BoundingBox,
    ContentOrigin,
    ElementType,
    ExtractionMethod,
    FileKind,
)
from omnirag.ingestion.ocr import OcrError, TesseractOcr

_MIN_OCR_CONFIDENCE = 0.3
_EXIF_ORIENTATION = 0x0112


class ImageLoader:
    name = "image"
    kinds = (FileKind.IMAGE,)

    def __init__(self, ocr: TesseractOcr | None = None) -> None:
        self.ocr = ocr

    def load(self, path: Path, ctx: ExtractionContext) -> dict[str, Any]:
        with Image.open(path) as img:
            orientation = img.getexif().get(_EXIF_ORIENTATION)
            dpi = img.info.get("dpi")
            meta = {
                "format": img.format,
                "width_px": img.width,
                "height_px": img.height,
                "mode": img.mode,
                "dpi": [round(float(v)) for v in dpi] if dpi else None,
                "exif_orientation": orientation,
            }
            upright = ImageOps.exif_transpose(img).convert("L")  # grayscale is enough for OCR

        image_id = ctx.add(
            ElementType.IMAGE,
            "",
            extraction_method=ExtractionMethod.PILLOW,
            asset_ref=ctx.original_ref,
            metadata=meta,
        )
        meta["ocr"] = self._ocr(upright, ctx, image_id, dpi)
        return meta

    def _ocr(
        self, upright: Image.Image, ctx: ExtractionContext, image_id: str | None, dpi: Any
    ) -> dict[str, Any]:
        if self.ocr is None or not self.ocr.available():
            ctx.warn("OCR unavailable: image text was not extracted")
            return {"performed": False}
        buffer = io.BytesIO()
        upright.save(buffer, format="PNG")
        try:
            # Only pass a DPI the file actually declares; a guess would mislead Tesseract.
            result = self.ocr.recognize(buffer.getvalue(), dpi=round(dpi[0]) if dpi else None)
        except OcrError as exc:
            ctx.warn(f"OCR failed: {exc}")
            return {"performed": False, "error": str(exc)}
        kept = [p for p in result.paragraphs if p.confidence >= _MIN_OCR_CONFIDENCE]
        for para in kept:
            x0, y0, x1, y1 = para.bbox
            ctx.add(
                ElementType.PARAGRAPH,
                para.text,
                extraction_method=ExtractionMethod.TESSERACT,
                origin=ContentOrigin.OCR,
                confidence=para.confidence,
                parent_id=image_id,
                bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, unit="px"),
                metadata={
                    "ocr_engine": self.ocr.version,
                    "words": para.words,
                    "low_confidence_words": para.low_confidence_words,
                },
            )
        return {
            "performed": True,
            "engine": self.ocr.version,
            "paragraphs": len(kept),
            "dropped_low_confidence": len(result.paragraphs) - len(kept),
            "mean_confidence": round(result.mean_confidence, 3),
            "duration_ms": result.duration_ms,
        }
