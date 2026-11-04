"""Loader registry: one loader per ``FileKind``."""

from __future__ import annotations

from omnirag.core.config import Settings
from omnirag.ingestion.loaders.base import Loader
from omnirag.ingestion.loaders.csv_loader import CsvLoader
from omnirag.ingestion.loaders.docx_loader import DocxLoader
from omnirag.ingestion.loaders.html_loader import HtmlLoader
from omnirag.ingestion.loaders.image_loader import ImageLoader
from omnirag.ingestion.loaders.json_loader import JsonLoader
from omnirag.ingestion.loaders.markdown_loader import MarkdownLoader
from omnirag.ingestion.loaders.pdf_loader import PdfLoader
from omnirag.ingestion.loaders.pptx_loader import PptxLoader
from omnirag.ingestion.loaders.text_loader import TextLoader
from omnirag.ingestion.loaders.xlsx_loader import XlsxLoader
from omnirag.ingestion.models import FileKind
from omnirag.ingestion.ocr import TesseractOcr


def default_loaders(settings: Settings | None = None) -> dict[FileKind, Loader]:
    """All loaders in this build. OCR is configured from ``settings`` (defaults when None)."""
    settings = settings or Settings(_env_file=None)
    ocr = (
        TesseractOcr(
            cmd=settings.ocr_tesseract_cmd,
            languages=settings.ocr_languages,
            timeout_s=settings.ocr_timeout_s,
        )
        if settings.ocr_enabled
        else None
    )
    loaders: list[Loader] = [
        TextLoader(),
        MarkdownLoader(),
        HtmlLoader(),
        JsonLoader(),
        CsvLoader(),
        DocxLoader(),
        XlsxLoader(),
        PptxLoader(),
        PdfLoader(ocr, ocr_dpi=settings.ocr_dpi, ocr_max_pages=settings.ocr_max_pages),
        ImageLoader(ocr),
    ]
    return {kind: loader for loader in loaders for kind in loader.kinds}
