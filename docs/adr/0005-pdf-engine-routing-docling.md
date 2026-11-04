# ADR-0005: Route PDFs between PyMuPDF and Docling ("auto"), Docling on CPU

**Status:** accepted (2026-11-04, Phase 2d)

## Context
PyMuPDF's heuristics (ADR-0004) are fast but fail on two common layouts, both measured on `data/samples/e7_technical_note.pdf`:
- **Two columns.** The right column's "3. Results" came out *before the document title*, and the numbered sections were jumbled (4, 1, 2).
- **Borderless tables.** A 6×3 table became six loose lines (`NaDFOB SEI additive 1 wt%`) with no column names.

Docling (layout + TableFormer models) reads both correctly. But it costs ~3.4 s for a 1-page note against 0.09 s with PyMuPDF (the 3-page safety review: 7.1 s vs 0.17 s, about 40×). It also uses ~1.2–1.45 GB RAM, a 506 MB one-time model download and ~60 s for the first pipeline initialisation. It flattened every heading to level 1, while PyMuPDF's font-size levels were right.

## Decision
- `OMNIRAG_PDF_ENGINE=auto` (default). PyMuPDF's first pass detects **multi-column pages**: two prose blocks of at least 40 characters side by side with vertical overlap. Only those documents go to Docling; everything else stays on PyMuPDF. `pymupdf` and `docling` force one engine.
- **Scanned pages always stay on PyMuPDF + Tesseract.** Docling's OCR is off, so we keep one OCR path with per-paragraph confidence (ADR-0004).
- If Docling fails, its partial output is dropped, PyMuPDF processes the document, and a warning is recorded.
- **Docling runs on CPU.** The GPU is reserved for Ollama. `torch`/`torchvision` are listed directly in the `layout` dependency group, pinned to PyTorch's CPU index. uv applies `[tool.uv.sources]` only to *direct* dependencies, and the first install silently pulled the CUDA build (`2.15.0+cu132`, 13 NVIDIA packages, a 5.9 GB venv). The fix gave `2.15.0+cpu` and 1.6 GB.
- Docling lives in a **default dependency group** (`layout`), so `uv sync` installs it and `uv sync --no-group layout` gives a slim install. Without it, `auto` silently uses PyMuPDF and `docling` warns.
- Pinned `docling==2.141.0` (16 days old at decision time); the project releases almost daily.

## Consequences
- Simple PDFs keep PyMuPDF speed. Complex layouts get model quality at a measured price.
- The detector can't see a **borderless table on a single-column page**. Such documents need `pdf_engine=docling` (a known gap, listed in the learning doc).
- Docling images are **rendered crops** of the page region, not the embedded stream; the original PDF is still kept byte-for-byte.
- First use downloads models from Hugging Face. Offline or Docker deployments must pre-fetch them (Phase 15).
- Real-Docling tests are marked `docling` and excluded from the default run, like `integration`.

## Alternatives considered
| Option | Why not |
|---|---|
| Docling for every PDF | About 40× slower on simple PDFs and worse heading levels, for no gain there |
| `pymupdf_layout` (PyMuPDF's layout add-on) | Same AGPL licence question; not evaluated yet, a candidate for a later comparison |
| Marker / Nougat | GPU-oriented, heavier; the GPU is reserved for Ollama |
| Column detection + our own re-ordering in PyMuPDF | Fixes order but not borderless tables; reinvents a layout model |
