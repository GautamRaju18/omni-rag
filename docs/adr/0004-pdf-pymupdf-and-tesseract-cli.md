# ADR-0004: PDF parsing with PyMuPDF; OCR through the Tesseract CLI

**Status:** accepted (2026-11-04, Phase 2c)

## Context
PDFs need fast text extraction with positions (for citations and bounding boxes), table detection, image extraction and page rendering for OCR. Scanned pages and image files need OCR with confidence scores. Everything has to run locally inside an 8 GB WSL VM.

## Decision
1. **PyMuPDF 1.28.2** (pinned with `==`) for PDF text, fonts, `find_tables`, image extraction and rendering.
   - 2.0 and 2.1 were released 2 and 16 days before this decision. We stay on the newest 1.x until 2.x has a track record; the golden tests are the safety net for the later upgrade.
2. **Tesseract 5.5.0 via its command line** (`subprocess`, no shell, image on stdin, TSV on stdout) instead of the `pytesseract` wrapper. It's one call, we need the TSV word confidences anyway, and it means one less dependency.

## Consequences
- **License: PyMuPDF is AGPL-3.0** (or a commercial license from Artifex). That's fine for this open-source portfolio project. A closed-source or SaaS deployment would need a commercial license or a swap to **pypdfium2** (BSD/Apache) plus pdfplumber for tables. The loader interface (`Loader.load(path, ctx)`) keeps that swap local to one file.
- PyMuPDF prints an advert for `pymupdf_layout` to stdout; the loader calls `pymupdf.no_recommend_layout()`.
- Tesseract must be installed on the host (`apt install tesseract-ocr`) and later in the Docker image (Phase 15). Without it, scanned pages and images are still stored, and a warning says OCR was skipped.
- Layout-heavy PDFs (multi-column, borderless tables, formulas) are only partly handled. Phase 2d evaluates **Docling** for those.

## Alternatives considered
| Option | Why not (now) |
|---|---|
| pypdf | Text only, no positions or table detection; slow on large files |
| pdfplumber / pdfminer.six | Good tables and positions, but much slower than PyMuPDF |
| pypdfium2 | Permissive license and fast, but no table detection: the fallback if the AGPL becomes a problem |
| Docling only | Best layout quality but heavy (ML models, GB of RAM); added in 2d for hard cases, not as the default |
| PyMuPDF built-in OCR (`get_textpage_ocr`) | Needs `TESSDATA_PREFIX` setup and returns no per-word confidence |
| EasyOCR / PaddleOCR | GPU-hungry; the 6 GB VRAM is reserved for Ollama models |
