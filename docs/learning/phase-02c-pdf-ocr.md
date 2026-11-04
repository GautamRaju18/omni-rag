# Phase 2c: PDF, scanned documents and images (PyMuPDF + Tesseract OCR)

> Builds on [2a](phase-02a-ingestion-core.md) and [2b](phase-02b-office-loaders.md). Decision record: [ADR-0004](../adr/0004-pdf-pymupdf-and-tesseract-cli.md).

## 1. Problem being solved
PDFs are the most common and the most hostile format for RAG:
- **A PDF has no paragraphs or headings.** It is drawing instructions ("glyph *A*, font F, size 11, at x,y"). Structure must be inferred from font sizes, positions and lines.
- **Every page repeats noise:** "Page 3 of 12", "Company Confidential". Left in, that noise appears in every chunk and pollutes retrieval.
- **Scanned PDFs have no text at all.** A page is just a photo of paper, so without OCR the document is invisible to search.
- **Some text layers are garbage.** Fonts without a Unicode map extract as `����`.
- **Photos from phones are stored sideways**, with an EXIF tag saying "rotate when displaying".

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Born-digital vs scanned** | Born-digital pages have a text layer. Scanned pages are an image with (almost) no text. Detection rule: fewer than 25 characters of text *and* images covering ≥ 50% of the page. |
| **Font-size heading inference** | Pass 1 finds the **body size**, the size covering the most characters. Short blocks at least 15% larger become headings, levels 1–3 by size rank. Checking font size *before* list markers keeps "1. Incident Summary" a heading. |
| **Boilerplate removal** | Text in the top/bottom 8% of the page that repeats on ≥ 50% of pages (digits masked, so "Page 1" = "Page 2") is a running header or footer and is skipped. |
| **PDF points and bbox** | 1 pt = 1/72 inch. Every element carries a `bbox` in points, and OCR boxes are converted from pixels (`px × 72 / dpi`). In Phase 11, a citation can highlight the exact region. |
| **OCR** | Optical character recognition. Tesseract's TSV output gives each word's text, box and **confidence (0–100)**. We group words into paragraphs, keep line breaks, and store the mean confidence as the element's `confidence`, with `origin=OCR`. |
| **DPI for OCR** | The page is rendered at `OMNIRAG_OCR_DPI` (default 300). Tesseract is trained on text roughly 20–40 px tall; too low loses detail and too high costs time (see §8). |
| **EXIF orientation** | A JPEG tag that tells viewers to rotate the image. Tesseract ignores it, so we apply `ImageOps.exif_transpose` first. |
| **Decompression bomb** | A small file that decodes to gigabytes of pixels. Pillow refuses images above ~179 MP when opening. PDFs are capped at 2,000 pages and 50 OCR pages per document (configurable). |

## 3. Architecture
```
PdfLoader(ocr, ocr_dpi, ocr_max_pages)
  pass 1: body font size, heading sizes, boilerplate keys      (whole document)
  pass 2, per page:
     text chars < 25 and image coverage ≥ 50%  ──► scanned ─┐
     U+FFFD ratio > 10%                         ──► garbled ─┤
                                                             ▼
                      IMAGE (scan, original bytes) ◄── render page @300 dpi ──► TesseractOcr
                      PARAGRAPH × n (origin=OCR, confidence, bbox pt, parent = IMAGE)
     otherwise: find_tables → TABLE/ROWs;  blocks → HEADING / LIST / PARAGRAPH / CAPTION;
                images → IMAGE assets (each xref stored once per document)

ImageLoader(ocr)
  IMAGE (asset_ref = stored original, no copy) ─► exif_transpose → grayscale → TesseractOcr
  PARAGRAPH × n (origin=OCR, bbox px, parent = IMAGE)

TesseractOcr: subprocess [tesseract stdin stdout -l eng --psm 3 --dpi N tsv], timeout, no shell
```
Settings (all `OMNIRAG_*`, see `.env.example`): `OCR_ENABLED`, `OCR_TESSERACT_CMD`, `OCR_LANGUAGES` (validated, e.g. `eng+deu`), `OCR_DPI`, `OCR_TIMEOUT_S`, `OCR_MAX_PAGES`.

Smaller changes:
- `ExtractionContext.original_ref` lets an image file be its own asset.
- `add_table(..., bbox=)`.
- `default_loaders(settings)` builds OCR from configuration.

## 4. Design decisions
- **Images are stored as found.** JPEGs inside a PDF come back **byte-identical**. PNGs don't exist inside PDFs: they're stored as compressed pixels, so extraction produces a new PNG with **identical pixels** but different bytes. A test asserts both.
- **OCR text links to its source image** (`parent_element_id`), so a citation can show the scan the text came from.
- **Low-confidence OCR fragments (< 30%) are dropped** with a warning. **Confidence alone is not a safety net, though:** in the EXIF experiment (§8), OCR on the sideways photo produced pure garbage at **34%** mean confidence, which is above the cutoff. Fixing the input (rotation) is what helped, not filtering the output.
- **OCR line breaks are kept.** In notebooks and forms each line is often a separate fact. Tesseract's grouping put the AUR-C105 and AUR-C106 lines into one paragraph, and joining them with spaces made them read as one sentence.
- **Without Tesseract nothing crashes.** Scans and images are still stored as IMAGE elements, and a warning explains that OCR was skipped. Phase 2d adds vision-model descriptions for images that contain no text.

## 5. File map
| File | Purpose |
|---|---|
| `src/omnirag/ingestion/ocr.py` | `TesseractOcr`, `parse_tsv`, `OcrResult` |
| `src/omnirag/ingestion/loaders/pdf_loader.py` | Two-pass PDF loader with scan/garbled detection |
| `src/omnirag/ingestion/loaders/image_loader.py` | PNG/JPEG loader with EXIF handling + OCR |
| `scripts/make_pdf_samples.py` | Generates the three samples below |
| `data/samples/aurora_safety_review.pdf` | Born-digital, 3 pages: numbered headings, ruled table, figure + caption, list, header/footer |
| `data/samples/aurora_lab_notebook_scan.pdf` | One scanned page (tilted 0.7°, dust, blur), no text layer |
| `data/samples/whiteboard_2026-03-20.jpg` | Sideways pixels + EXIF orientation 6 |
| `tests/unit/ingestion/test_pdf_image_ocr.py` | 15 tests (5 need the tesseract binary and skip without it) |

New pinned dependency: `pymupdf==1.28.2`. System dependency: `tesseract-ocr` 5.5.0 (apt).

## 6. How to run
```bash
cd /mnt/p/omni-rag && uv sync
uv run python -m omnirag.ingestion ingest data/samples
uv run python -m omnirag.ingestion show doc_3b75ba9ed290ce86          # the scanned notebook
OMNIRAG_OCR_ENABLED=false uv run python -m omnirag.ingestion ingest --force data/samples/whiteboard_2026-03-20.jpg
```

## 7. How to test
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                                          # 136 unit tests
uv run pytest tests/unit/ingestion/test_pdf_image_ocr.py -v
```

## 8. Expected outputs (captured 2026-11-04: PyMuPDF 1.28.2, Tesseract 5.5.0)
```text
COMPLETED  doc_3b75ba9ed290ce86   pdf           5    756.4  aurora_lab_notebook_scan.pdf
COMPLETED  doc_2b1c659c54266d54   pdf          19    174.2  aurora_safety_review.pdf
COMPLETED  doc_db47e30200f6a13a   image         5    215.5  whiteboard_2026-03-20.jpg
```
The other 8 samples are `DUPLICATE`, since nothing changed for them.

Safety review (born-digital): `boilerplate_removed: ['helios energy lab - internal safety review', 'page # of #']`
```text
[02] heading    p1  1. Incident Summary                      (level 2, 15 pt; body is 11 pt)
[08] table_row  p2  section: Safety Review > 3. Swelling Measurements   conf 0.85
     Cell: AUR-C103 | Electrolyte: E-5 | Temp (°C): 45 | Cycles: 412 | Swelling (%): 14.0
[14] caption    p2  parent = [13] image (page002_xref28.png)
     Figure 1: Capacity retention at the latest checkpoint, AUR-C101 to AUR-C108.
```
Scanned notebook (OCR at 300 dpi):
```text
[00] image      p1  assets/page001_xref5.jpeg   bbox 0,0,612,792
[02] paragraph  p1  origin=ocr  conf 0.91  parent=[00]  words 24, low-confidence 0
     AUR-C105 (E-7, 45 C): 600 cycles, retention 89.6 %, swelling 2.1 %
     AUR-C106 (E-7, 45 C): 600 cycles, retention 90.2 %, swelling 2.4 %
```
**EXIF experiment** (same photo, same Tesseract):
| Input | Paragraphs | Mean confidence | Text |
|---|---|---|---|
| Raw pixels (EXIF ignored) | 3 | **0.34** | `Oo1e/\| JEUMO / (CW) S89 OOL -YOIeq…` (garbage) |
| `exif_transpose` (upright) | 4 | **0.96** | `E-7 QUALIFICATION PLAN / 1. Swelling limit: 5 % / …` |

**DPI experiment** (notebook page, 7 key facts):
| DPI | OCR time | Mean conf | PNG size | Facts read |
|---|---|---|---|---|
| 150 | 254 ms | 0.934 | 269 KB | 7/7 |
| 200 | 300 ms | 0.927 | 473 KB | 7/7 numbers correct, but `2.1%` lost its space, and a stray `\|` appeared in a wide gap |
| 300 | 455 ms | 0.919 | 931 KB | 7/7 |
| 400 | 669 ms | 0.930 | 1434 KB | 7/7 |

My first scoring script said "200 dpi failed" because it compared exact strings. That's a lesson for Phase 12: **normalize whitespace before scoring OCR or extraction output**, or you'll measure formatting instead of correctness. 300 dpi is the default: it read every fact cleanly, and the cost is ~0.5 s per page.

Cost per file (CLI process total including Python start-up): safety review 1.41 s / 113 MB RSS; scanned page 2.30 s / 135 MB; whiteboard 1.40 s / 116 MB. The loaders themselves take 174 / 756 / 216 ms. Data dir: 828 KB for all 11 samples.

## 9. Bugs and surprises found during verification
1. **Numbered headings parsed as list items.** "1. Incident Summary" matched the list-marker rule before the font-size check, and "5. Approval" was swallowed into the bullet list above it. Fixed by checking font size first, and a mutation test proves the golden test catches it.
2. **OCR merged separate lines into one sentence.** Fixed by keeping line breaks.
3. **PyMuPDF printed an advert to stdout.** It's silenced with its own `no_recommend_layout()` switch, found by reading the library source.
4. **My test assumed PNGs survive PDF round-trips byte-for-byte.** They don't (see §4). The test now checks pixels for PNG and bytes for JPEG.
5. **A Phase 2a test used "PDF has no loader" as its example.** That's no longer true: the test now builds a pipeline without the PDF loader, and a new test checks that a corrupt PDF becomes `FAILED` with the original kept.

## 10. Common errors
| Symptom | Cause | Fix |
|---|---|---|
| `⚠ Page N looks scanned ... was not OCR'd` | Tesseract missing, `OCR_ENABLED=false`, or the `OCR_MAX_PAGES` limit | `sudo apt install tesseract-ocr`, or raise the limit |
| `FAILED ... PDF is password-protected` | Encrypted PDF | Decrypt first: `qpdf --decrypt --password=… in.pdf out.pdf` |
| `Tesseract failed: ... Failed loading language 'deu'` | Language pack missing | `sudo apt install tesseract-ocr-deu`, then `OMNIRAG_OCR_LANGUAGES=eng+deu` |
| OCR text is garbage at ~30–40% confidence | Rotated or upside-down scan, or very low resolution | Check `exif_orientation`; rescan at ≥ 300 dpi. Page-rotation detection (Tesseract `--psm 0`) is a possible extension |
| Headings missing in a PDF | Document uses bold at body size for headings | Known limit of size-based inference; Docling (2d) uses a layout model |
| Two columns read interleaved | Reading order is top-to-bottom | Known limit; Docling in 2d |
| `DecompressionBombError` | Image over ~179 MP | Expected protection; downscale the image first |

## 11. Exercise (≈45 min)
Add **automatic page-rotation detection** for scans fed into the scanner sideways.
1. Tesseract's orientation detection: `tesseract stdin stdout --psm 0` prints `Rotate: 90` and `Orientation confidence: …` (it needs the `osd` traineddata, already installed).
2. Add `TesseractOcr.detect_rotation(image: bytes) -> int` that parses `Rotate:` from that output.
3. In `PdfLoader._ocr_page`, if the rotation is not 0 and the confidence is above 2, rotate the pixmap (Pillow `Image.rotate(-angle, expand=True)`) before `recognize`, and record `metadata.rotation_corrected`.

Acceptance: a test that builds a scanned PDF page rotated by 90° (reuse `_scanned_pdf`, rotate the image first), sees the OCR output contain "89.6", and finds `rotation_corrected == 90`; ruff is clean.

## 12. Acceptance criteria — Phase 2c
- [x] PDF: text with headings (levels by font size), sections across pages, lists, ruled tables, images with captions, bbox in points, page numbers
- [x] Running headers/footers removed; password-protected and corrupt PDFs fail cleanly with the original kept
- [x] Scanned-page detection plus OCR with confidence, `origin=OCR`, bbox, and a link to the scan image
- [x] Garbled-text-layer detection routes to OCR
- [x] Images: original kept (no copy), EXIF rotation applied, OCR text linked to the image
- [x] Works without Tesseract: content kept, explicit warnings
- [x] 136 unit tests + 4 integration tests pass; mutation checks confirm the key tests can fail
