# Phase 2a — Ingestion core + text-family loaders

> Status: **complete and verified on 2026-11-04.** 83 unit tests (59 new), Phase 1 integration tests still green, sample corpus ingested.
>
> Phase 2 is split into checkpoints: **2a** core + TXT/MD/HTML/JSON/CSV · **2b** DOCX/XLSX/PPTX · **2c** PDF (PyMuPDF) + Tesseract OCR + images · **2d** Docling + vision-model descriptions · **2e** `/documents` and `/ingestion` API with background jobs.

## 1. Problem being solved
Every file format represents content differently: a Markdown heading, an HTML `<h2>`, a JSON key, a CSV header row. If retrieval code had to understand formats, every new format would touch every layer. Ingestion therefore turns **any file into one list of `DocumentElement`s**: typed, normalized, ordered units with full provenance. Everything after this phase depends only on that schema.

Ingestion must also be **safe** (reject disguised or malformed files), **idempotent** (re-uploading the same file is not new data), and **lossless** (the original is always kept, so a better parser later can re-process it).

## 2. Concepts
| Concept | Where | Why |
|---|---|---|
| **Unified element schema** | `models.py` | Adapter pattern: each loader adapts one format to one shared output. |
| **Content-addressed IDs** | `validation.py` | `document_id = "doc_" + sha256[:16]`. The same bytes always get the same ID, so duplicates are detected by a lookup before any parsing. Renaming a file doesn't fool it. |
| **Deterministic element IDs** | `context.py` | `<doc_id>-e00042` = position in reading order. Re-ingesting yields identical IDs, so evaluation ground truth that references them stays valid. Random UUIDs would break the Phase 12 dataset every time you re-ingest. |
| **Magic bytes** | `validation.py` | Extensions are claims. `%PDF-`, `\x89PNG`, `\xFF\xD8\xFF`, and the internal ZIP layout of Office files (`word/document.xml` …) are facts. |
| **Content origin** | `models.py` | `original` (verbatim), `ocr` (may contain recognition errors), `generated` (model-written, never a quote). Citations in Phase 11 must treat these differently. |
| **Confidence** | every loader | 1.0 = certain. Lowered for guesses: cp1252 encoding fallback (0.8), an HTML table without `<th>` (×0.8), heuristic CSV headers (×0.95). OCR will use engine confidences. |
| **Unicode normalization** | `normalize.py` | NFC + ligature/space/invisible-char fixes. **Not NFKC**, which would turn `Na₂` into `Na2` and `x²` into `x2`. |
| **Atomic writes** | `store.py` | Write to a temp file, then `os.replace`, which is atomic on POSIX. A crash never leaves a half-written `document.json`. |
| **Failure isolation** | `pipeline.py` | A loader exception produces a `FAILED` record with the error. The original is kept, other files continue, and a later run retries it. |

## 3. Architecture and data flow
```
path ──► validate_file ──► sha256 ──► document_id ──► store.read_record
           │ ext allowlist   (streamed,                   │ COMPLETED & !force → DUPLICATE (no work)
           │ size limit       1 MiB chunks)               ▼
           │ magic bytes                         save_original (before parsing!)
           │ OOXML layout                        record.status = PROCESSING
           ▼                                              ▼
   UnsupportedFileType (415)          loader.load(path, ExtractionContext)
   FileValidation (422)                 └─ ctx.add(...) → normalize → skip empty → deterministic ID
   LoaderNotAvailable (501)                                ▼
                                       elements.jsonl + document.json (COMPLETED | FAILED)
```
Storage layout (`OMNIRAG_DATA_DIR`, default `~/omnirag-data` on WSL ext4):
```
documents/doc_72ec135197a4981b/
├── original.md       byte-for-byte copy of the upload
├── document.json     DocumentRecord: status, sha256, loader, counts, metadata, warnings, error
├── elements.jsonl    one DocumentElement per line
└── assets/           (2b+: embedded images, page renders)
```

## 4. Loaders and their design decisions
| Loader | Library | Key decisions |
|---|---|---|
| `TextLoader` | stdlib | Paragraphs split on blank lines. **No heading guessing**, because a false `section_path` is worse than none. Line numbers kept as the citation anchor. Single newlines are kept (in plain text they often matter: `Date:` / `Attendees:`). |
| `MarkdownLoader` | markdown-it-py (CommonMark) | Walks top-level block tokens. `#` inside code fences is not a heading (tested). Heading stack gives `section_path`. Soft line breaks become spaces. Front matter is blanked rather than removed, so line numbers stay exact. Images keep `src` and are **not fetched**. |
| `HtmlLoader` | BeautifulSoup (`html.parser`) | Removes script/style/nav/footer/aside/forms, then uses `<main>` → `<article>` → `<body>`. A recursive block walker emits headings, paragraphs, nested lists, `<pre>`, tables and figures (caption linked to its image). Remote images are not fetched (SSRF risk, offline-first). |
| `JsonLoader` | stdlib | One `RECORD` per object with its scalar fields as `key: value` lines. Short scalar lists are inlined. Nested objects recurse as children. `metadata.json_path` = `$.team[1]`. Per-file state is passed as arguments, not stored on the shared loader (thread safety for the API). |
| `CsvLoader` | stdlib `csv` | `Sniffer.sniff` detects the delimiter. Header detection is a **type rule first** (unique non-numeric names above a numeric column), with `Sniffer.has_header` as a fallback (see bug 4 below). |

**Tables (shared `tables.py`).** One `TABLE` element (Markdown, keeps structure for the LLM) plus one `TABLE_ROW` child per row rendered as `Header: value | Header: value`. A bare row `| AUR-C103 | 45 | 412 |` is meaningless to an embedding model. `Cell: AUR-C103 | Temp (°C): 45 | Cycles: 412` can be retrieved and cited by itself. The same helper will serve DOCX, XLSX, PPTX and PDF tables.

**Alternatives considered.** *Unstructured.io* gives an element schema for many formats, but it's heavy (many system deps), its output is less controllable, and it hides exactly what this project is meant to teach. *Trafilatura* is better than BeautifulSoup at main-content extraction from arbitrary web pages and is worth trying if HTML quality becomes a measured problem. *pandas* for CSV is overkill for reading; it arrives with XLSX in 2b.

## 5. File map
```
src/omnirag/ingestion/
├── __init__.py            build_pipeline(settings)
├── __main__.py            CLI: ingest / list / show
├── models.py              DocumentElement, DocumentRecord, enums
├── normalize.py           normalize_text()
├── validation.py          validate_file(), sha256_file(), document_id_for()
├── context.py             ExtractionContext.add() / save_asset() / warn()
├── tables.py              add_table(), to_markdown(), row_to_text()
├── store.py               DocumentStore (file-based, atomic, traversal-safe)
├── pipeline.py            IngestionPipeline.ingest()
└── loaders/
    ├── base.py            Loader protocol, read_text() encoding fallback
    ├── text_loader.py  markdown_loader.py  html_loader.py  json_loader.py  csv_loader.py
    └── __init__.py        default_loaders() registry
data/samples/              fictional "Project Aurora" corpus (5 files, cross-referenced facts)
tests/unit/ingestion/      normalize, validation, loaders, pipeline, golden sample-corpus tests
```
`core/errors.py` gained `FileValidationError` (422), `UnsupportedFileTypeError` (415), `LoaderNotAvailableError` (501) and `NotFoundError` (404). `core/config.py` gained `data_dir` and `max_upload_mb`.

## 6. How to run
```bash
cd /mnt/p/omni-rag
uv run python -m omnirag.ingestion ingest data/samples          # README* files are skipped
uv run python -m omnirag.ingestion ingest data/samples --force  # re-process
uv run python -m omnirag.ingestion list
uv run python -m omnirag.ingestion show doc_72ec135197a4981b --limit 20
```

## 7. How to test
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                                  # 83 unit tests, no services
uv run pytest tests/unit/ingestion -v          # just this phase
```

## 8. Expected outputs (captured 2026-11-04)
```text
STATUS     DOCUMENT ID            KIND      ELEMS       MS  FILE
COMPLETED  doc_add7b677937e9d7a   json          9      5.5  aurora_config.json
COMPLETED  doc_72ec135197a4981b   markdown     19      7.6  aurora_research_notes.md
COMPLETED  doc_244886254b9fdfa5   csv           9      5.7  cell_test_results.csv
COMPLETED  doc_8c8d2a4b69ee5cee   html         15      9.8  helios_aurora_s1.html
COMPLETED  doc_a6e2ce2a78478e8a   txt           6      4.4  meeting_notes_2026-03-12.txt
```
Second run without `--force`: every row says `DUPLICATE`, with no parsing work done.

Rejections (exit code 1):
```text
REJECTED   tool.exe: Unsupported file type '.exe'
REJECTED   report.pdf: report.pdf has a .pdf extension but no PDF signature
REJECTED   paper.pdf: No loader available for pdf files yet
```
Element examples:
```text
[e00013] table_row  section: Project Aurora: Research Notes > Results (as of 2026-03-10)
         Cell: AUR-C103 | Electrolyte: E-5 | Temp (°C): 45 | Cycles: 412 | Retention (%): 72.5
[e00002] record     section: team[1]   (json_path $.team[1], parent = root record)
         name: Marco Lindqvist ⏎ role: Electrolyte chemist ⏎ workstreams: electrolyte, E-7 qualification
[e00012] caption    parent = e00011 (image "Aurora S1 pouch cell next to a ruler for scale")
         Figure 1: Aurora S1 pilot-line cell, 2026.
```
The HTML page's nav ("Careers"), cookie banner, footer ("Privacy") and `<script>` contribute **zero** elements (golden test).

## 9. Bugs found during verification
1. **Soft-wrapped paragraphs** (`... cell for ⏎ stationary storage`) came from the smoke run, not a test. In Markdown and HTML a single newline is formatting, so it's now joined. Plain text keeps newlines on purpose.
2. **Text line numbers off by one.** The paragraph regex consumed exactly one blank line, so with two blank lines the next block started with `\n`. It now consumes all blank lines and computes start/end from the stripped block.
3. **Nested list indentation lost.** `LIST` wasn't layout-sensitive, so normalization stripped the indentation. It now is, alongside `CODE` and `TABLE`.
4. **`csv.Sniffer.has_header` said "no header" for `id;value;note / A1;10;ok`.** Its string-length vote treats `id` and `A1` (both length 2) as the same kind of value. Fix: a type-based rule first, the Sniffer only as a fallback, and the method recorded as `header_detection_method`. Lesson: **standard-library heuristics are still heuristics. Test them on your own data.**
5. Two test bugs: `"\x80"` in a Python string is U+0080, not `€`; and a golden test matched the whole table instead of the row. Tests can be wrong too, so read the failure before "fixing" the code.

## 10. Common errors
| Symptom | Cause | Fix |
|---|---|---|
| `REJECTED ... no PDF signature` | Renamed or corrupt file | Expected; check the real type with `file <path>` |
| `No loader available for X yet` | Format arrives in 2b–2d | Expected until then |
| Everything is `DUPLICATE` | Same bytes ingested before | `--force`, or delete `~/omnirag-data/documents/<id>` |
| A `FAILED` record | Loader exception | `show <id>` prints `error`; the original is kept, so fix and re-run |
| `warnings: ["Not valid UTF-8; decoded as cp1252"]` | Legacy Windows encoding | Content is fine; confidence is lowered |

## 11. Exercise (≈45 min)
Add **`.jsonl` (JSON Lines)** support: one JSON object per line, common for logs and datasets.
1. Map `.jsonl` to `FileKind.JSON` in `validation.py`.
2. In `JsonLoader.load`, if the suffix is `.jsonl`, parse each non-empty line separately and emit each with `json_path = "$[<line_index>]"` and `metadata.line_number`.
3. A malformed line should add a warning and be skipped, without failing the whole file.

Acceptance: a unit test with 3 valid lines and 1 broken line yields 3 records and 1 warning; `ruff` is clean; existing tests still pass.

## 12. Acceptance criteria — Phase 2a
- [x] Unified `DocumentElement` with every required field: document_id, element_id, element_type, content, source_file, page_number, metadata, extraction_method, confidence, parent_element_id, asset reference (+ section_path, order, origin, bbox)
- [x] Original vs generated content distinguished (`origin`)
- [x] Originals preserved byte-for-byte before parsing
- [x] Duplicate detection by SHA-256 (rename-proof); `--force` re-ingest keeps element IDs identical
- [x] Validation: extension allowlist, size limit, magic bytes, OOXML layout, binary-in-text
- [x] Ingestion status: PROCESSING → COMPLETED / FAILED with error and warnings persisted
- [x] TXT, Markdown, HTML, JSON, CSV loaders with structure, provenance and confidence
- [x] Golden tests on a cross-referenced sample corpus
