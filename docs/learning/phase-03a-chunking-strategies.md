# Phase 3a: Chunking engine (seven strategies)

> Builds on Phase 2's elements. Decision record: [ADR-0008](../adr/0008-chunking-tokens-and-element-provenance.md). Phase 3b adds semantic chunking, a starter evaluation set and the retrieval benchmark.

## 1. Problem being solved
A retriever doesn't search documents; it searches **chunks**. Chunk boundaries decide what can be found together:
- **Too big:** the chunk contains the answer plus a lot of unrelated text. Its embedding is a blur, and the LLM reads noise.
- **Too small:** a fact is cut in half ("…retained" | "89.6% after 600 cycles"), or a table row loses its header.
- **Wrong boundaries:** two sections merged, a heading stranded at the end of a chunk, an image's caption separated from its OCR text.

There is no universally best strategy, so we build several and measure them on the same documents.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Token** | The unit the embedding model counts. We use **bge-m3's own tokenizer**; its counts plus 2 special tokens equal Ollama's `prompt_eval_count` exactly. |
| **Chunk size / overlap** | The maximum tokens per chunk, and how many tokens consecutive chunks share (so a fact at a boundary appears whole in at least one). |
| **Recursive splitting** | Try the coarsest boundary first (paragraph), then line, sentence, clause, word, then a hard token cut, then pack small pieces back up to the size. |
| **Structure-aware** | Use the document's own structure: sections and elements. Never cross a section, never split an element that fits. |
| **Parent–child** | Retrieve on small, precise children; give the LLM their larger parent as context. |
| **Hierarchical** | Several nested levels (2048 / 512 / 128), all searchable: a query can match a whole section or one sentence. |
| **Provenance** | Each chunk lists the `element_ids` it contains (and which only partly), so pages, sections, tables and figures stay traceable for citations and evaluation. |
| **Contextual headers** | An optional `context` line ("Document: … / Section: …") prepended **only to the embedding text**, never to the content a citation shows. |
| **Integrity** | The share of elements kept whole in at least one chunk: a retrieval-free proxy for "can a single chunk answer a question about this element?". |

## 3. The strategies
| Strategy | How it builds chunks | Tables | Images / charts |
|---|---|---|---|
| `fixed` | Exact token windows over the plain text, stepping `size − overlap` | cut anywhere | cut anywhere |
| `recursive` | Paragraph → line → sentence → clause → word → token splits, packed up to the size, with overlap | cut at line breaks | as text |
| `structure` | Whole elements packed per **section**; a heading starts a chunk and stays with its content; oversized elements are split recursively (marked `partial`) | inside the section's text | as separate texts |
| `table` | structure, plus every table is **its own chunk**; big tables are split **by rows, header repeated** in every part | own chunk(s) | as separate texts |
| `multimodal` | table, plus every image/chart is **one figure chunk**: caption + alt text + chart data + OCR text (with confidence) + generated description (with model name) | own chunk(s) | one labelled figure chunk |
| `hierarchical` | Level 0 = a whole **top-level section** (≤ 2048); level 1 ≤ 512; level 2 ≤ 128; no child identical to its parent | as multimodal | as multimodal |
| `parent_child` | Parents = multimodal chunks (≤ 512, **not indexed**); children ≤ 128 (indexed) | as multimodal | as multimodal |
| `semantic` | *Phase 3b*: boundaries where embedding similarity drops | | |

Every chunk carries the brief's fields: `chunk_id`, `document_id`, `source_file`, `page_number` (+ `page_numbers`), `section_path`, `element_type`, `chunk_index`, `content`, `token_count`, `metadata`, `embedding_ref` (filled in Phase 4), `parent_chunk_id` and `chunking_strategy`. It also carries `element_ids`, `partial_element_ids`, `level`, `retrievable`, `origins` (original/ocr/generated) and `min_confidence`.

## 4. Architecture
```
IngestionPipeline: … loader → enrichers → write elements → Chunker.chunk(record, elements)
                                                            └─ for each configured strategy:
                                                               chunks/<config_id>.jsonl
chunking/
  tokenizer.py   HFTokenizer (bge-m3, pinned + SHA-256 verified), WhitespaceTokenizer (tests)
  text.py        Assembled text + Segments (one per element), split_spans, token_windows
  strategies.py  _Builder: blocks → pack (per section) → emit (spans → element provenance)
  service.py     Chunker, integrity, describe (stats)
  __main__.py    CLI: compare | rechunk | show
```
The key design choice: **everything works on character spans into one assembled text**. A chunk is a span, and its elements are the segments the span overlaps. That's why even the 128-token children of a hierarchical chunk know exactly which table row or OCR paragraph they came from.

## 5. File map
| File | Purpose |
|---|---|
| `src/omnirag/chunking/models.py` | `ChunkingStrategy`, `ChunkingConfig` (+ validation, `config_id`), `Chunk` |
| `src/omnirag/chunking/tokenizer.py` | bge-m3 tokenizer download/verification; counters |
| `src/omnirag/chunking/text.py` | Span-based recursive splitter, fixed token windows |
| `src/omnirag/chunking/strategies.py` | The seven strategies |
| `src/omnirag/chunking/service.py` | `Chunker`, `integrity`, `describe`, `level_limit` |
| `src/omnirag/chunking/__main__.py` | CLI |
| `src/omnirag/ingestion/store.py` | + `write_chunks` / `read_chunks` / `chunk_config_ids`; re-ingest clears stale chunks |
| `tests/unit/chunking/` | 42 tests (whitespace tokens, no network) |
| `tests/integration/test_tokenizer_live.py` | Our counts = Ollama's |

New settings: `OMNIRAG_CHUNKING_ENABLED`, `OMNIRAG_CHUNKING_STRATEGIES` (JSON list), `OMNIRAG_CHUNK_SIZE` (512), `OMNIRAG_CHUNK_OVERLAP` (64), `OMNIRAG_CHUNK_CHILD_SIZE` (128), `OMNIRAG_CHUNK_PARENT_SIZE` (2048), `OMNIRAG_CHUNK_CONTEXTUAL_HEADERS`, and the pinned tokenizer repo/revision/SHA-256.

## 6. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
uv run python -m omnirag.chunking compare --size 128 --size 256 --size 512
uv run python -m omnirag.chunking rechunk --all          # after changing chunk settings
uv run python -m omnirag.chunking show doc_2b1c659c54266d54 table-512 --limit 8
uv run python -m omnirag.chunking show doc_865ac481a2392f72 multimodal-512
```

## 7. How to test
```bash
uv run ruff format --check . && uv run ruff check .
uv run pytest                                 # 218 unit tests
uv run pytest -m integration -k tokenizer     # bge-m3 counts vs Ollama
```

## 8. Expected outputs (captured 2026-11-05, 12 sample documents, bge-m3 tokenizer)
```text
CONFIG                       CHUNKS INDEXED   MED   P95   MAX OVER  INTACT      KB      MS
fixed-128-32                     50      50   128   128   128    0   53.6%    53.2    39.9
recursive-128-32                 53      53   104   124   125    0   77.7%    51.3    38.5
structure-128                    74      74    70   124   128    0   97.3%    63.3    30.0
table-128                        76      76    68   122   128    0   97.3%    63.5    29.0
multimodal-128                   82      82    68   120   128    0   95.3%    69.8    40.7
hierarchical-2048-128-64        175     175    52   200   677    0  100.0%   166.0    90.4
parentchild-128-64-32           225     143    43    61    64    0   95.3%   186.8    82.5
fixed-512-64                     16      16   294   512   512    0   99.0%    27.6    18.2
structure-512                    48      48    66   201   342    0  100.0%    46.1    17.8
multimodal-512                   55      55    49   246   342    0  100.0%    51.2    21.6
parentchild-512-128-64          132      77    65   118   124    0  100.0%   122.6    50.5
```
How to read it, without over-reading it:
- At 128 tokens, **fixed windows keep only 53.6% of elements whole** (recursive 77.7%, structure 97.3%). Fixed cuts sentences and table rows anywhere.
- At 512, fixed and recursive put most of our small documents into **one chunk each** (16 chunks for 12 files). That says more about this corpus than about the strategies; Phase 12's larger dataset should include longer documents.
- Structure-family chunks are **smaller** (median 49–70 tokens) because they stop at section ends. Smaller chunks may embed more precisely, or lose context. **Retrieval quality is not measured yet** (Phase 3b), and whole elements alone don't prove better answers.
- Multimodal at 128 has *lower* integrity (95.3%) than table-128 (97.3%), because a figure group with a long generated description exceeds 128 and gets split.
- Hierarchical indexes about 3× more chunks (all levels) and parent–child about 2.5× more chunk records, so index size and latency grow with them.

Example chunks from the safety review:
```text
[table-512] table 146 tok p=[2] section=… > 3. Swelling Measurements
    3. Swelling Measurements ⏎ | Cell | Electrolyte | Temp (°C) | Cycles | Swelling (%) | ⏎ …
[multimodal-512] figure 246 tok p=[2] origins=generated,original
    Figure: Figure 1: Capacity retention … ⏎ Image description (generated by qwen3-vl:2b-instruct): …
```
Costs: tokenizer load 2.3 s and **+302 MB RSS**; counting 35k characters takes 17 ms. Re-chunking all 12 documents with 7 strategies: 2.9 s (mostly the tokenizer load). Chunk files: ≤ 80 KB per document.

## 9. Bugs found during verification
1. **Hierarchical was identical to multimodal** (55 chunks each). The top level grouped by exact section, so 2048 was never reached, and parents that fitted 512 got no leaves at all. Fixed: the top level groups a whole top-level section (`pack(depth=1)`), and leaves are created from any chunk bigger than the child size.
2. **Re-tokenizing a slice ≠ the slice of tokens.** 9 of 50 fixed windows at 128 tokens came out as 129. Windows are now trimmed until the slice itself fits.
3. **Orphan headings.** Overlap carried "1. Incident Summary" (5 tokens) into a window of its own, and recursive splits stranded headings at the end of a window. Fixed: windows never end right after a heading (`keep_with_next`), and windows already covered by the previous one are dropped.
4. **Misleading `element_type`.** A fixed window that merely touched a table was labelled "table", and a chart whose data rows outweigh its title was labelled "table". Now the label is the character majority, and figure/table groups name themselves.
5. **A config that ignored overlap was rejected** (`chunk_size=64` with the default overlap 64). A default that's too large is now scaled to `size // 4`; an explicit bad overlap is still an error.
6. **A test gap found by a mutation check:** allowing children identical to their parent didn't fail any test, because the test document had no tiny section. One was added; the mutation now fails.
7. Test bugs: a child size below the schema minimum, a miscounted window total, and an expectation that recursive splitting never packs small paragraphs together (packing up to the size is the point).

## 10. Common errors
| Symptom | Cause | Fix |
|---|---|---|
| `⚠ Chunking failed: TokenizerDownloadError` | Offline on first use | Connect once, then `python -m omnirag.chunking rechunk --all` |
| `Checksum mismatch for …tokenizer.json` | Wrong or changed file upstream | Never bypass: check the pinned revision/SHA-256 in settings |
| `chunk_overlap must be smaller than chunk_size` | An explicit overlap ≥ size | Lower `OMNIRAG_CHUNK_OVERLAP` |
| `hierarchical needs child_size < chunk_size < parent_size` | Inconsistent level sizes | Fix the three sizes |
| `No chunks for … (available: …)` in `show` | Wrong config ID, or the document was ingested before 3a | `rechunk --all`, then use an ID from the list |
| API memory grew ~300 MB | Tokenizer loaded on first chunking | Expected (ADR-0008) |

## 11. Exercise (≈45 min)
Add a **`sentence` strategy**: chunks of whole sentences, packed up to the size, never crossing a section.
1. Reuse `split_spans` with separators `(". ", "? ", "! ", "\n")` on each section's assembled text.
2. Register it in `ChunkingStrategy` and `chunk_document`, with `config_id` `sentence-<size>`.
3. Compare it with `recursive` and `structure` at 128 tokens using `compare --strategy …`.

Acceptance: a unit test showing no chunk ends mid-sentence on a three-paragraph document; `compare` shows its INTACT and median tokens next to the others (record your numbers and date); ruff is clean.

## 12. Acceptance criteria — Phase 3a
- [x] Seven strategies (fixed, recursive, structure-aware, table-aware, multimodal, hierarchical, parent–child), with configurable size, overlap, child and parent sizes
- [x] Token-aware with the embedding model's own tokenizer (verified against Ollama)
- [x] All required chunk fields, plus element-level provenance and origins
- [x] Optional contextual headers (embedding text only)
- [x] Chunk sets stored per configuration; re-ingest clears stale chunks; chunking failure is a warning
- [x] Structural comparison measured on the sample corpus (no retrieval claims)
- [x] 218 unit + 7 live tests pass; ruff clean
- [ ] Phase 3b: semantic chunking, LLM contextual enrichment experiment, evaluation set, Recall@K / MRR / nDCG benchmark
