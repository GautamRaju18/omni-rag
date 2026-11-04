# Phase 2d: Layout model (Docling) and image descriptions (vision model)

> Builds on [2c](phase-02c-pdf-ocr.md). Decisions: [ADR-0005](../adr/0005-pdf-engine-routing-docling.md) (PDF engine routing) and [ADR-0006](../adr/0006-vision-model-image-descriptions.md) (vision model).

## 1. Problem being solved
After 2c, two kinds of content were still lost or mangled:
1. **Complex PDF layouts.** In a two-column paper, PyMuPDF's top-to-bottom ordering mixed the columns: a result appeared *before the title*. A table without ruling lines became loose lines of text, with no column names.
2. **What images show.** OCR reads text in an image but can't say "this bar chart shows retention per cell, with C103 the outlier at 72.5". A chart in a report was just an IMAGE with an empty `content`.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Layout model** | A vision model that classifies page regions (title, text, list, table, picture, page header…) and their reading order from the rendered page, instead of font-size rules. Docling uses one, plus **TableFormer** for table structure. |
| **Furniture** | Docling's name for page headers and footers. They are recognised visually per page and excluded from the body, with no repetition rule needed. |
| **Engine routing** | A cheap detector decides per document whether the expensive engine is worth it. `auto`: multi-column → Docling, otherwise PyMuPDF; scanned pages always go to PyMuPDF + Tesseract. |
| **Vision-language model (VLM)** | A model that takes an image plus text and writes text. We run it locally via Ollama with base64 images on `/api/chat`. |
| **Generated vs original content** | A description is the model's interpretation. It is stored as its own `DESCRIPTION` element with `origin=generated`, a policy confidence of 0.5 and a link to its IMAGE. It can help answer questions, but must never be cited as a quote. |
| **Enrichment step** | Runs after the loader and only **appends** elements. Loader element IDs therefore never depend on whether vision is on, off or failing, and Phase 12's evaluation ground truth stays valid. |
| **Model residency** | Ollama keeps a model in VRAM for `keep_alive`. The VLM stays loaded for a batch and is unloaded once at the end (`pipeline.close()`), not per document. |

## 3. Architecture
```
PdfLoader
  pass 1 (PyMuPDF): fonts, boilerplate, multi_column_pages, scan_like_pages
  _choose_engine:  pymupdf (configured / single-column / scans need OCR / docling missing)
                   docling (configured / multi-column)  ──► DoclingEngine.convert()
                                                               lazy converter, lock, CPU torch
                                                               body items → TITLE/HEADING/LIST/TABLE/IMAGE/CAPTION
                                                               failure → drop partial output, fall back to PyMuPDF
IngestionPipeline.ingest(): loader → enrichers (append only) → persist
  ImageDescriptionEnricher: IMAGE with asset, unless OCR ≥ 30 words
       → OllamaVision.describe(prepare_image(bytes), hint = alt text + caption)
       → DESCRIPTION (origin=generated, confidence 0.5, parent = IMAGE)
IngestionPipeline.close(): unload the VLM after the batch (CLI calls it in `finally`)
```
New settings (`.env.example`): `OMNIRAG_PDF_ENGINE=auto|pymupdf|docling`, `OMNIRAG_VISION_ENABLED`, `OMNIRAG_VISION_MODEL`, `OMNIRAG_VISION_TIMEOUT_S`, `OMNIRAG_VISION_MAX_IMAGES`.

## 4. Design decisions (and the measurements behind them)
- **Docling only where it pays.** On the 3-page safety review it took 7.1 s against PyMuPDF's 0.17 s, and it flattened headings to level 1. On the two-column note it was the only engine that got the order and the table right. Hence `auto`.
- **Docling on CPU.** The GPU belongs to Ollama. Getting CPU-only torch took a second attempt (§9, bug 1).
- **qwen3-vl:2b over larger models.** On our images it read 24/24 facts and paired all 8 chart labels with their values, at 0.9–1.7 s per image and 2.13 GB VRAM. qwen3-vl:4b paired **0/8**, and gemma3:4b was about 2× slower. Bigger was not better here.
- **Prompt rules:** transcribe exactly, never guess, at most 150 words; temperature 0, fixed seed. The prompt has a version (`describe-v1`), stored on every description, because changing the wording changes the output.
- **Text images are skipped:** the scanned notebook already has 51 OCR words, so a description would only paraphrase it at a cost of seconds.

## 5. File map
| File | Purpose |
|---|---|
| `src/omnirag/ingestion/loaders/docling_engine.py` | Lazy, locked Docling converter + mapping to our elements |
| `src/omnirag/ingestion/loaders/pdf_loader.py` | + multi-column/scan detection and engine routing |
| `src/omnirag/ingestion/vision.py` | `OllamaVision`, `prepare_image`, `ImageDescriptionEnricher` |
| `src/omnirag/ingestion/pipeline.py` | `Enricher` protocol, enrichment step, `close()` |
| `scripts/benchmark_vision.py` | Reproducible VLM comparison (facts, pairs, invented values, timing, VRAM) |
| `data/samples/e7_technical_note.pdf` | Two columns + borderless table (generated by `make_pdf_samples.py e7_technical_note.pdf`) |
| `tests/unit/ingestion/test_vision.py` | 8 tests with a fake Ollama (`httpx.MockTransport`) |
| `tests/unit/ingestion/test_pdf_engines.py` | Routing with a fake engine + 1 real-Docling test (`-m docling`) |
| `tests/integration/test_vision_live.py` | The real VLM reads the retention chart (`-m integration`) |

Dependencies: the `layout` group = `docling==2.141.0`, `torch==2.15.0` (CPU), `torchvision==0.29.2` (CPU). Ollama model: `qwen3-vl:2b-instruct` (1.93 GB download).

## 6. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
ollama.exe pull qwen3-vl:2b-instruct        # or from PowerShell: ollama pull ...
uv run python -m omnirag.ingestion ingest --force data/samples
uv run python -m omnirag.ingestion show doc_efcae56deb2b7631       # the two-column note (Docling)
OMNIRAG_VISION_ENABLED=false uv run python -m omnirag.ingestion ingest --force data/samples/whiteboard_2026-03-20.jpg
uv run python scripts/benchmark_vision.py qwen3-vl:2b-instruct     # re-measure on your machine
```

## 7. How to test
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                        # 147 fast tests (~7 s)
uv run pytest -m "integration or docling"   # 6 tests: Neo4j, Ollama, vision model, real Docling (~20 s warm)
```

## 8. Expected outputs (captured 2026-11-04)
Two-column note with `pdf_engine=auto`: `engine: docling, engine_reason: multi-column pages [1]`
```text
[00] heading    E-7 Electrolyte: Technical Note
[01] paragraph  Abstract. E-7 is a carbonate electrolyte ...
[02] list       1. Motivation ... ⏎ 2. Formulation ... ⏎ 3. Results ... ⏎ 4. Conclusion ...   (in order)
[03] table      conf 0.90, detector "docling tableformer"
[08] table_row  Component: NaDFOB | Role: SEI additive | Amount: 1 wt%
```
The same file with PyMuPDF: "3. Results" was element 0 (before the title), the list read 4, 1, 2, and the table was six loose paragraphs.

A description (safety review, page 2, `origin=generated`, parent = the chart IMAGE):
```text
This image is a bar chart titled "Figure 1: Capacity retention at the latest checkpoint, AUR-C101 to AUR-C108." It displays data for eight categories, ...
```
Full sample batch (`ingest --force data/samples`, 13 files including README skipping): **24.9 s**, max RSS 1.24 GB. Four images described (1.3–2.2 s each once loaded; the first call paid a 4.2 s load), one skipped (the scanned notebook), and the VLM unloaded afterwards (GPU back to 1048 MiB). The same batch with per-document unloading took 40.9 s.

Element-ID stability: the status report ingested with vision on and off differs only by the extra `doc_b55492ddcf5fc0c8-e00019 description` line.

## 9. Bugs and surprises found during verification
1. **CUDA torch installed despite the CPU index.** uv ignores `[tool.uv.sources]` for transitive dependencies, so torch came from PyPI as `2.15.0+cu132` (13 NVIDIA packages, a 5.9 GB venv) and `torch.cuda.is_available()` was `True`: Docling would have used the GPU. Fix: list `torch`/`torchvision` directly in the `layout` group. Result: `2.15.0+cpu`, 1.6 GB. My first check ("no CUDA packages installed") was itself broken. **Verify with the value that matters** (`torch.__version__`, `cuda.is_available()`), not a grep that can fail silently.
2. **A 154 s outlier in the first VLM benchmark.** The model was competing with the already-loaded chat model. The benchmark now unloads everything before each model and runs each image twice. One run is not a benchmark.
3. **Scoring artifact.** qwen3-vl:4b wrote "C101 to C108", so a per-label check called five labels "missing". The real problem was different and only showed up with a better metric: it never paired labels with values. **Design the metric around what retrieval needs.**
4. **Temperature 0 is not determinism.** The first call after loading differed in wording from later identical calls (same facts). Descriptions are therefore never exact-match ground truth.
5. **Per-document unloading cost 6–7 s per image.** Moving the unload to the end of the batch: 41 s → 25 s.
6. Docling and transformers print "Loading weights" progress bars into CLI output. Silenced with `transformers.utils.logging.disable_progress_bar()`.

## 10. Common errors
| Symptom | Cause | Fix |
|---|---|---|
| `⚠ Image description skipped: ... 404` | VLM not pulled | `ollama pull qwen3-vl:2b-instruct` |
| `⚠ Image description skipped: ... ConnectError` | Ollama not running | Start Ollama; the document is still stored |
| First Docling document takes ~60 s | Model download (506 MB) and pipeline initialisation | One-time; later starts are about 10 s |
| `⚠ pdf_engine=docling but Docling is not installed` | `uv sync --no-group layout` was used | `uv sync` |
| Very slow image descriptions, GPU shows other models | VLM competing for VRAM with the chat model | Ingest batches when chat isn't in use; check `/api/v1/health/ollama` |
| `torch.cuda.is_available()` is `True` in this venv | CUDA wheel came back after editing deps | Keep `torch`/`torchvision` in the `layout` group (ADR-0005) |
| Borderless table on a single-column page not detected | `auto` only routes multi-column documents | `OMNIRAG_PDF_ENGINE=docling` for such files |

## 11. Exercise (≈60 min)
Make `auto` also route **single-column PDFs with borderless tables** to Docling.
1. In `_Plan`, look for runs of ≥ 3 consecutive short blocks (< 40 characters each) whose left edges line up in at least 2 columns (x0 values within 3 pt across rows), that don't sit inside a `find_tables()` rectangle.
2. Record such pages as `table_like_pages` and route to Docling in `_choose_engine`, with reason `"borderless table on pages [...]"`.
3. Guard against false positives: the safety-review PDF (ruled table) must stay on PyMuPDF.

Acceptance: a fast unit test with a single-column page containing a borderless table (reuse the HTML from `make_pdf_samples.BORDERLESS_TABLE`) is routed to a `FakeEngine`; the safety review still reports `engine: pymupdf`; ruff is clean.

## 12. Acceptance criteria — Phase 2d
- [x] Docling integrated on CPU; reading order and borderless tables correct on the two-column sample; furniture excluded
- [x] `auto` routing with recorded `engine` and `engine_reason`; scanned PDFs stay on OCR; Docling failure falls back cleanly
- [x] Vision model chosen by a reproducible benchmark on our own images; numbers recorded with versions
- [x] Image descriptions stored as `origin=generated` children of their images; loader IDs unchanged with vision on or off
- [x] VLM unloaded after each batch; works (with warnings) when Ollama or the model is missing
- [x] 147 unit + 5 integration + 1 Docling tests pass; ruff clean
