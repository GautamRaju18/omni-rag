# Phase 2b: Office loaders (DOCX, XLSX, PPTX)

> Builds on [Phase 2a](phase-02a-ingestion-core.md). Every loader still emits `DocumentElement`s only through `ExtractionContext.add()`, and tables go through `tables.add_table()`.

## 1. Problem being solved
Most company knowledge lives in Word reports, Excel workbooks and PowerPoint decks. These are not text files. Each is a **ZIP archive of XML parts** (Office Open XML, "OOXML"), and the meaning sits in the structure:

- In Word, a heading is a paragraph with an outline level, a list item is a paragraph with numbering properties, and a reviewer's edit is wrapped in `<w:ins>` / `<w:del>`.
- In Excel, a "table" is just a rectangle of cells somewhere on a grid. `0.725` with the format `0.0%` means **72.5%**, and `46172` with a date format means **2026-05-30**.
- In PowerPoint, XML order is drawing order, not reading order. A chart's numbers sit in the file as data, and the speaker notes often hold the facts the presenter actually says.

Flatten any of this to a text dump and retrieval loses headings, units, row context and provenance.

## 2. Concepts
| Concept | What it means here |
|---|---|
| **OOXML** | `.docx`, `.xlsx` and `.pptx` are ZIPs. `word/document.xml`, `xl/worksheets/sheet1.xml` and `ppt/slides/slide1.xml` hold the content; `media/` holds the images. Phase 2a's validator already checks these markers. |
| **Zip bomb** | A small archive that declares gigabytes of uncompressed content. The validator now sums the declared sizes from the ZIP directory (nothing is decompressed) and rejects anything over 512 MiB. Python's `zipfile` stops reading a member at its declared size, so the declared size can't be exceeded later. |
| **XML bombs / XXE** | Entity-expansion attacks ("billion laughs") hidden in the XML. python-docx and python-pptx parse with `resolve_entities=False` (checked in their source). openpyxl uses **defusedxml** when it is installed, so it is now a dependency, and a test asserts `openpyxl.DEFUSEDXML is True`. |
| **Outline level** | `w:outlineLvl` (0–8) marks a Word heading regardless of the UI language. Style names are localized ("Heading 1" is "Überschrift 1" in German Word), so the name is only a fallback. |
| **Style inheritance** | A paragraph property (outline level, numbering) can be set on the paragraph or inherited from its style, which inherits from its `base_style`. The loader walks that chain. |
| **Tracked changes** | `<w:ins>` holds proposed insertions, and `<w:del>` holds deletions as `w:delText`. We extract the text **as if the changes were accepted** and record a warning, so the user knows. |
| **Cached values vs formulas** | An `.xlsx` cell stores the formula *and* the last value Excel calculated. Files written by scripts (openpyxl, pandas) have formulas but **no cached value**. |
| **Read-only mode** | `openpyxl.load_workbook(read_only=True)` streams rows instead of building every cell object, which keeps memory flat for big workbooks. |
| **EMU** | PowerPoint positions are in English Metric Units (914,400 per inch). python-pptx's `Length.pt` converts them, and that gives slide `bbox` coordinates in points. |

## 3. Architecture
```
validate_file ── ZIP layout marker ✓  declared uncompressed size ≤ 512 MiB ✓
      │
      ▼
DocxLoader / XlsxLoader / PptxLoader  (stateless, shared across threads)
      │   ctx.add(...)       → HEADING / PARAGRAPH / LIST / CAPTION / FIGURE / IMAGE
      │   add_table(...)     → TABLE + self-describing TABLE_ROWs (+ row_metadata)
      │   ctx.save_asset(...)→ documents/<id>/assets/image_001.png (original bytes)
      ▼
DocumentStore (unchanged from 2a)
```
Two shared-layer changes:
- `normalize_text` now turns `\v`, `\f`, U+2028 and U+2029 into newlines. Before, the control-character filter *deleted* them, which would have turned PowerPoint's `line\vbreak` into `linebreak`.
- `add_table(..., row_metadata=[...])` attaches per-row provenance, such as `{"excel_row": 7}`.

## 4. Loaders and their design decisions

### DOCX: `loaders/docx_loader.py` (python-docx 1.2.0)
- **Own text extraction (the most important decision in this phase).** python-docx's `Paragraph.text` is `"".join(e.text for e in p.xpath("w:r | w:hyperlink"))`, which reads direct children only. Text inside `<w:ins>`, `<w:fldSimple>`, smart tags and inline content controls disappears without an error. `_element_text` reads every `w:t` inside a run and skips anything under `w:del`, `w:moveFrom`, text boxes and `mc:Fallback` (the duplicate legacy copy of a drawing). The test `test_python_docx_paragraph_text_misses_insertions` documents the upstream behaviour. If python-docx ever fixes it, that test fails and tells us to revisit.
- **Headings** come from the outline level (paragraph, then style chain), then a "Heading N" name. The `Title` style becomes a `TITLE` element.
- **Lists** come from `w:numPr` (paragraph or style). `numFmt` from the numbering part decides between `-` and `1.`. Nested items render indented. Built-in styles like "List Bullet 2" nest by style name rather than `ilvl`, so that's handled too.
- **Tables:** `row.cells` repeats a merged cell for every grid column it spans, so merged headers still label every column. A declared header row (`w:tblHeader`, "Repeat as header row") gives confidence 1.0, an inferred one 0.9. **Single-row or single-column tables are treated as layout tables** (Word users often use tables for layout) and emitted as paragraphs.
- **Images:** each `a:blip r:embed` is saved as an asset with its alt text (`wp:docPr/@descr`). A following `Caption`-style paragraph becomes a `CAPTION` whose parent is that image. Linked images (`r:link`) are **not fetched** (no SSRF).
- **Skipped for now:** headers and footers (boilerplate, verified absent in the golden test), footnotes, comments, text boxes, charts, cover pages and tables of contents.

### XLSX: `loaders/xlsx_loader.py` (openpyxl 3.1.5)
- **Two read-only passes in lockstep:** cached values (`data_only=True`) and formulas. If a cell has a formula but no cached value, the formula text (`=SUM(B2:B5)`) is emitted, the block's confidence is ×0.8, and a warning is raised. An empty cell would make the model believe "there's no total".
- **Display rendering** (`render_value`): percentage formats are respected (`0.0%` gives `72.5%`), dates come out as ISO, booleans as `TRUE`/`FALSE`, and whole floats lose their `.0`. Other number formats use the raw value, which is more precise than the display.
- **Grid → blocks:** split at fully empty rows and trim empty edge columns. A block with ≥2 rows and ≥2 columns becomes a table; anything else becomes a paragraph (titles, notes). `reset_dimensions()` plus `min_row=1, min_col=1` make grid indices equal Excel coordinates, so `cell_range` ("B4:E8") and `excel_row` are exact citations.
- **Header rule:** row 1 is the header if its names are unique, non-empty and non-numeric (confidence 0.95). Otherwise `column_N` names are generated (0.85). Typed cells make this simpler than CSV.
- Hidden sheets are ingested, and their `sheet_state` is recorded. Empty sheets produce nothing. There's a guard of 100,000 rows per sheet.

### PPTX: `loaders/pptx_loader.py` (python-pptx 1.0.2)
- One slide = one page. The slide title becomes a `HEADING` and the section name; untitled slides get `Slide N`.
- **Reading order:** shapes are sorted by `(top, left)`, and groups are walked recursively. Top-level shapes get a `bbox` in points.
- Body placeholders become a `LIST` indented by `paragraph.level`. A soft break inside a bullet joins with a space. Other text boxes become a `PARAGRAPH`. Footer, date and slide-number placeholders are skipped.
- **Charts become data:** a `FIGURE` ("Chart: <title>", chart type, series names) with a child `TABLE` built from the chart's cached categories and values. The golden test checks that the deck's chart (`AUR-C105: 89.6` at 600 cycles) agrees with the CSV, so the same fact is now retrievable from two formats.
- **Speaker notes** become a `PARAGRAPH` with `speaker_notes: true` and section `[title, "Speaker notes"]`.
- Pictures are saved as assets with their alt text. Hidden slides are kept and flagged (`hidden: true`, listed in `hidden_slides`). SmartArt and OLE objects are counted in a warning.

## 5. File map
| File | Purpose |
|---|---|
| `src/omnirag/ingestion/loaders/docx_loader.py` | Word loader + `_element_text` |
| `src/omnirag/ingestion/loaders/xlsx_loader.py` | Excel loader + `render_value` (also used for chart values) |
| `src/omnirag/ingestion/loaders/pptx_loader.py` | PowerPoint loader |
| `src/omnirag/ingestion/validation.py` | + zip-bomb guard (`MAX_OOXML_UNCOMPRESSED_BYTES`) |
| `src/omnirag/ingestion/normalize.py` | + line-separator fix |
| `src/omnirag/ingestion/tables.py` | + `row_metadata` |
| `scripts/make_office_samples.py` | Generates the 3 Office samples (documented, reproducible structure) |
| `data/samples/aurora_status_report_2026-03.docx`, `aurora_budget_2026.xlsx`, `aurora_q1_review.pptx` | Committed samples (source of truth) |
| `tests/unit/ingestion/test_office_loaders.py` | Builds tiny documents in-test, one structure per test |
| `tests/unit/ingestion/test_sample_corpus.py` | + 7 golden tests, including cross-format consistency |

New pinned dependencies: `python-docx==1.2.0`, `openpyxl==3.1.5`, `python-pptx==1.0.2`, `defusedxml==0.7.1`. These pull in `lxml 6.1.3`, `pillow 13.0.0`, `XlsxWriter 3.3.1` and `et-xmlfile 2.0.0`.

## 6. How to run
```bash
cd /mnt/p/omni-rag
uv sync                                                    # installs the new dependencies
uv run python -m omnirag.ingestion ingest data/samples
uv run python -m omnirag.ingestion show doc_865ac481a2392f72 --limit 30   # the deck
uv run python scripts/make_office_samples.py               # only to regenerate (new IDs!)
```

## 7. How to test
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                                   # 117 unit tests
uv run pytest tests/unit/ingestion/test_office_loaders.py -v
```

## 8. Expected outputs (captured 2026-11-04)
```text
STATUS     DOCUMENT ID            KIND      ELEMS       MS  FILE
COMPLETED  doc_7126e7e76ad7b561   xlsx         17     39.8  aurora_budget_2026.xlsx
DUPLICATE  doc_add7b677937e9d7a   json          9      5.5  aurora_config.json
COMPLETED  doc_865ac481a2392f72   pptx         25     76.2  aurora_q1_review.pptx
DUPLICATE  doc_72ec135197a4981b   markdown     19      7.6  aurora_research_notes.md
COMPLETED  doc_b55492ddcf5fc0c8   docx         19     69.4  aurora_status_report_2026-03.docx
           ⚠ Document contains tracked changes (1 insertions, 1 deletions); extracted text shows them as if accepted
DUPLICATE  ... (csv, html, txt from Phase 2a)
```
Element examples:
```text
docx  paragraph  section: Summary
      Room-temperature cycling remains on track. The E-5 baseline failed at 45 °C (AUR-C103),
      so high-temperature testing moved to electrolyte E-7.        ← sentence 2 is a tracked insertion
xlsx  table_row  section: Budget   excel_row: 3   (table cell_range A1:E6)
      Workstream: Electrolyte | Owner: Marco Lindqvist | Planned (EUR): 240000 | Spent to date (EUR): 97200 | Spent (%): 40.5%
xlsx  table      section: Suppliers   cell_range: B4:E8           ← offset table found
pptx  table_row  page 3, parent = chart table, whose parent = FIGURE "Chart: Capacity retention (%) vs. cycles, E-7 at 45 °C"
      Category: 600 | AUR-C105: 89.6 | AUR-C106: 90.2
pptx  paragraph  page 2, section: Highlights > Speaker notes
      Mention that the steering committee approved an extra EUR 85,000 for the E-7 qualification on 2026-03-19.
```
Assets stored: `documents/doc_b55492ddcf5fc0c8/assets/image_001.png` and `documents/doc_865ac481a2392f72/assets/slide005_shape3.png`. The data dir is 292 KB in total.

Measured cost per Office file (CLI process, including Python start-up): **1.8–2.0 s wall, 72–77 MB max RSS**. The loader itself takes 40–76 ms; the rest is interpreter and import time.

## 9. Bugs and gaps found during verification
1. **python-docx drops tracked insertions** (found by reading its source before writing code). Fixed with `_element_text`; a test pins the upstream behaviour.
2. **`normalize_text` deleted `\v`** and glued words together. It's a shared-layer bug that would also have hit PDFs (`\f`). Fixed for all formats.
3. **A bullet list with a nested item was split in two.** "List Bullet 2" uses its own `numId`, and the rule "new numId means new list" fired on the way back to level 0. Now only a level-0 item whose numId differs from the list's *first* item starts a new list. This was found in the smoke run, not by a test; a test now covers it.
4. **Mutation check found a test gap.** With the `\v` fix removed on purpose, the normalize tests failed but the PowerPoint test still passed, because bullets replace `\v` themselves. A speaker-notes assertion now covers the non-bullet path. Lesson: **a green test only means something if it can go red.** Break the code on purpose once.
5. The sample `.docx` reported a "modified" date of 2013, inherited from python-docx's built-in template. Real files are unaffected; the generator now sets it.

## 10. Common errors
| Symptom | Cause | Fix |
|---|---|---|
| `REJECTED ... expands to N MiB when unpacked` | Zip-bomb guard | Expected for hostile files; raise `MAX_OOXML_UNCOMPRESSED_BYTES` only for trusted huge workbooks |
| `⚠ formula cells have no cached value` | File saved by a script, not Excel | Open and save it in Excel/LibreOffice, or accept the formula text |
| `⚠ tracked changes` | Unaccepted revisions in the Word file | Content shows them as if accepted; accept or reject in Word for exact text |
| `⚠ Skipped a drawing without an embedded picture` | Word chart, SmartArt or shape | Expected until Phase 2d |
| `.doc` / `.xls` / `.ppt` rejected | Legacy binary formats, not OOXML | Convert: `libreoffice --headless --convert-to docx file.doc` |
| `.xlsm` rejected | Macro-enabled workbook | Not supported on purpose (macros); save as `.xlsx` |
| A FAILED docx with `KeyError` | Corrupt relationship part | `show <id>` gives the error; the original is kept |

## 11. Exercise (≈45 min)
Extract **Word footnotes**. They often carry citations and caveats.
1. In `DocxLoader.load`, get the footnotes part with `document.part.part_related_by(RT.FOOTNOTES)` (catch `KeyError` when the document has none).
2. For each `w:footnote` with a numeric `w:id` ≥ 1 (ids −1 and 0 are separators), take the text of its paragraphs with `_element_text`.
3. Emit a `PARAGRAPH` with `metadata={"footnote_id": id}` and `section_path=["Footnotes"]`.

Acceptance: a unit test that builds a document with one footnote. python-docx has no API for adding footnotes, so inject the part, or **ask me to help build the fixture**. The test asserts one footnote element and that the separator footnotes are ignored; `ruff` is clean.

## 12. Acceptance criteria — Phase 2b
- [x] DOCX: title, headings by outline level, section paths, nested/numbered lists, tables with header confidence, images saved with alt text, captions linked, tracked changes handled and reported
- [x] XLSX: every sheet, blocks and offsets, Excel-style percentages and dates, uncached-formula detection, `cell_range` and `excel_row` provenance, empty sheets skipped
- [x] PPTX: slides as pages, reading order, bullets with levels, tables, charts as data tables, pictures as assets, speaker notes, hidden slides flagged
- [x] Zip-bomb guard and XML-bomb protection (defusedxml) verified by tests
- [x] Golden tests: the three samples ingest; facts agree across formats (budget docx ↔ xlsx, chart pptx ↔ csv)
- [x] 117 unit tests + 4 integration tests pass; ruff clean
