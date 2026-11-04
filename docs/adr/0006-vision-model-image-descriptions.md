# ADR-0006: Image descriptions with qwen3-vl:2b-instruct, stored as generated content

**Status:** accepted (2026-11-04, Phase 2d)

## Context
Charts, diagrams and photos hold facts that text extraction and OCR can't express. A local vision-language model (VLM) can describe them, but its output is generated text, not the document's words. The 6 GB GPU already holds the chat model (3.08 GB) and the embedding model (0.64 GB).

## Measurement (2026-11-04, Ollama 0.40.14, `scripts/benchmark_vision.py`, no document hints)
Three images with known content: the retention bar chart (8 values, 8 labels), the pouch diagram (1 label) and the sideways whiteboard photo (7 facts). Each model starts from an empty GPU; each image runs twice.

| Model | Facts read | Chart label-value pairs | Invented values | Cold load | Warm per image | VRAM |
|---|---|---|---|---|---|---|
| **qwen3-vl:2b-instruct** | 24/24 | **8/8** | 0 | 5.2 s | 0.9–1.7 s | **2.13 GB** |
| gemma3:4b | 24/24 | 8/8 | 0 | 7.0 s | 2.4–4.7 s | 2.80 GB |
| qwen3-vl:4b-instruct | 19/24 | **0/8** | 0 | 6.1 s | 2.1–2.6 s | 3.76 GB |

The 4B Qwen model listed all eight values but never said which cell each belonged to, so a question like "what is C105's retention?" can't be answered from it. A first run showed 154 s for that model on the chart. That was GPU contention with the already-loaded chat model, not the model itself, so models now start from an empty GPU.

**Three images are a sanity check, not a general ranking.** Re-run the script on new kinds of images before relying on the choice.

## Decision
- Default `OMNIRAG_VISION_MODEL=qwen3-vl:2b-instruct`. Prompt `describe-v1`: transcribe text and numbers exactly, never guess, at most 150 words. Settings: temperature 0, seed 42, longest image side 1280 px.
- Descriptions are separate `DESCRIPTION` elements: `origin=generated`, child of their IMAGE, `confidence=0.5`. That 0.5 is a **policy value meaning "unverified"**, not a measurement.
- Description runs as an **enrichment step after the loader**, appending elements. Loader element IDs are identical with vision on, off or failing (verified with a diff).
- Images whose OCR already produced ≥ 30 words (scanned text pages) are skipped.
- The VLM stays loaded for an ingestion **batch** and is unloaded by `pipeline.close()`. Unloading per document made every described image cost 6–7 s; per batch they cost 1.3–2.2 s after the first. The full sample ingest dropped from 41 s to 25 s.

## Consequences
- **Not byte-reproducible.** With temperature 0 and a fixed seed, the first call after loading differed in wording from later calls (same facts). Later calls were identical; the likely cause is prompt-cache reuse in Ollama. Descriptions must never be exact-match ground truth in evaluation.
- Generated text can still be wrong. Phase 11 citations must point to the IMAGE, never quote a description as the document's words.
- If Ollama or the model is unavailable, the document still completes, with a warning.
