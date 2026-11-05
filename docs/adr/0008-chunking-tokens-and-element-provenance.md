# ADR-0008: Chunk in embedding-model tokens; every chunk keeps element-level provenance

**Status:** accepted (2026-11-05, Phase 3a)

## Context
The brief asks for 8 chunking strategies, token-aware sizes, configurable overlap, and a benchmark across strategies on one evaluation dataset. Two questions decide whether that comparison is fair:
1. **What is a "token"?** Every strategy must be measured in the same unit as the embedding model's limit.
2. **What is "relevant"?** Chunk IDs differ per strategy and size, so ground truth written as chunk IDs would have to be rewritten for every configuration.

## Decision
- **Sizes are bge-m3 tokens**, counted with bge-m3's own `tokenizer.json` (Hugging Face `BAAI/bge-m3`, revision `5617a9f6…`, SHA-256 `21106b6d…`), downloaded once into `<data_dir>/models/` and verified.
  - It matches Ollama exactly: our count + 2 special tokens = Ollama's `prompt_eval_count` on 5 of 5 test strings. A live test keeps checking this.
  - A characters-per-token estimate would be wrong for this corpus: 3.2 for prose, 2.1 for table rows, 1.4 for `Na₂Fe[Fe(CN)₆]`, and 2.93 on the research notes.
- **Every chunk records `element_ids`** (plus `partial_element_ids` for elements it only partly contains), computed from character spans rather than by searching for strings. Ground truth is written once, as **element IDs** (deterministic, ADR content-addressing). A chunk is relevant if it contains a gold element. That works for every strategy and size, and for every future change of chunk sizes.
- **Every configured strategy runs at ingestion**: they take milliseconds (all 7 on 12 documents: about 20–90 ms). Chunk sets are stored per configuration (`chunks/<config_id>.jsonl`, e.g. `table-512`). Which set gets embedded is decided in Phase 4.
- Chunk IDs are `<document_id>-<config_id>-<index>`: deterministic for a given configuration.

## Consequences
- Loading the tokenizer costs ~2.3 s and **~300 MB RSS** (bge-m3's 250k-token vocabulary). That's affordable in the 8 GB VM, but it's resident in the API process. Phase 4 reuses the same instance.
- First use needs one 17 MB download from huggingface.co (a fixed URL from settings, never from document content). Offline or Docker deployments must pre-fetch it (Phase 15). If it's unavailable, ingestion still completes, with a "Chunking failed" warning, and `python -m omnirag.chunking rechunk --all` catches up later.
- Re-tokenizing a slice of text can give one token more than the slice of tokens it was cut from: 9 of 50 fixed windows at 128 came out as 129. Fixed windows are trimmed until the slice itself fits.

## Alternatives considered
| Option | Why not |
|---|---|
| Characters or words as the size unit | Off by up to 2× on tables and formulas compared with the model's real limit |
| `tiktoken` (OpenAI tokenizers) | A different vocabulary from bge-m3 |
| Ollama for counting (`/api/embed` → `prompt_eval_count`) | Computes a full embedding just to count, about 2.5 s on a cold model; good for verification only |
| LangChain / LlamaIndex text splitters | Return strings, not spans, so element-level provenance would need string search; plus a large dependency for ~150 lines of logic |
| Ground truth as chunk IDs | Invalid as soon as a strategy or size changes |
