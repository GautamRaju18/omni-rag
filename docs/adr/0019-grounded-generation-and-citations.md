# ADR-0019: Claims with verbatim quotes, checked by code; refusal without a model call; SQLite traces

**Status:** accepted (2026-11-06, Phases 10-11)

## Context
Phase 9c hands the generator a bounded, numbered context (`[1] file > section\ntext`). The brief asks the answer to use only that evidence, cite document names and pages, say when the evidence is insufficient, separate direct evidence from inference, handle several documents, tables and image descriptions, and map **each claim to source chunk IDs** without ever fabricating a citation. It also asks for streaming, structured output, timeouts, retries, and a chat history.

A 4B model writes fluent text whatever its context says, and "[2]" after a sentence is just two characters it can type. So a citation can't be trusted because the model wrote it.

## Decision
**1. The answer is a list of claims** (one structured call, `generation/prompts.py`, `PROMPT_VERSION = answer-v1`). Each claim has:
- `text`: the claim;
- `sources`: the block numbers it rests on;
- `quote`: words copied from one of those blocks;
- `support`: `direct` or `inferred`.

The reply also says `answerable` and `missing`. The sources come first in the prompt and the question last.

**2. Code verifies every claim** (`generation/citations.py`):
- Source numbers outside the context are dropped, and the drop is recorded.
- The quote is normalised (NFKC, dashes, case, whitespace, markdown) and searched in **every** block:
  - in a cited block → `verified`;
  - only in another block → `relocated` (the citation is corrected to where the words are, and flagged).
- A quote may leave out columns of one real table row: `Category: 300 | AUR-C106: 96.2` matches the line `Category: 300 | AUR-C105: 95.9 | AUR-C106: 96.2`. The rule is "elided", and its parts must appear in order *within one line*.
- **Every number in the claim must occur in its cited blocks.** A true quote with a changed number in the paraphrase ("swelled by 15%" quoting "14%") drops to `partial`.
- With no quote match, the claim is:
  - `partial` if its numbers, identifiers and most of its content words occur in the cited blocks;
  - otherwise `unsupported`.
- `direct` needs a verified or relocated quote. A quote found *only* inside a generated image description makes the claim `inferred` (`generated_evidence`). Text that is also in OCR or alt text counts as original.
- **Unsupported claims are removed** from the answer text but kept in the response (hallucination detection). An answer whose claims are all unsupported counts as a refusal.
- Every citation carries `source_file`, `section_path`, `page_numbers`, `chunk_ids`, `element_ids` and `origins` from the block.

**3. No context, no call.**
- If the context check (Phase 9a) found a question identifier in no passage (`insufficient`), the generator refuses: "No source mentions AUR-C109."
- An empty context is refused too.

Both take 0 model calls. `respect_check=false` turns this off (measured in Phase 12).

**4. Streaming at claim granularity** (`POST /chat/stream`, SSE). Ollama streams the JSON token by token. `ClaimStream` cuts out each claim object as soon as its closing brace arrives (it tracks strings and escapes), so the client sees *validated* claims one by one. The final `answer` event is parsed from the complete reply and is authoritative.

**5. Errors never fail a turn.**
- Model failures become `refused_by="error"` and the turn is still stored.
- Neo4j or embedding failures during retrieval are 503s, as in `/retrieval/advanced`.

**6. Chat history and traces in SQLite** (`generation/history.py`, `<data_dir>/omnirag.db`). One row per turn holds the full query lifecycle as JSON:
- question and strategy;
- queries, route and agent steps;
- every ranked list, with rerank scores;
- the selected context;
- the answer and its citations;
- per-component latency.

PostgreSQL (the brief's suggestion "when the application reaches that phase") was not adopted:
- the API is one process by design (ADR-0007);
- the data is kilobytes per turn;
- the 8 GB VM already holds Neo4j.

The SQL is plain, and the store is one class, so moving to Postgres for a multi-process deployment is a contained change.

## Measured (2026-11-06, qwen3:4b-instruct, default workflow; [live outputs in the learning doc](../learning/phase-10-generation.md))
| Question | Result |
|---|---|
| Why did cell AUR-C103 fail? | 2 claims, both verified (research notes, meeting notes); 3.6 s generation, 1,319 → 196 tokens |
| Which electrolyte performs better at 45 °C, E-5 + FEC or E-7? | 8 claims from 3 documents, all verified; 15.3 s (718 output tokens) |
| What does the whiteboard photo say about the pilot batch? | 1 claim verified against the OCR line |
| AUR-C106 retention after 300 cycles | 96.2%. The first run was `partial` (the quote dropped a column), which led to the elided-quote rule, unit-tested on that exact line |
| What capacity retention did cell AUR-C109 reach? | refused by the check, 0 model calls, 26 ms end to end |
| Who supplies the salt that the E-5 electrolyte is made with? | refused by the model: the supplier row wasn't in the context that run (the agent took a graph step instead), so it said what was missing instead of guessing |

The 110-question evaluation is in ADR-0020.

## Consequences
- **The answer can be shorter than what the model wrote:** unsupported claims are cut. That is the intended trade.
- **Citation accuracy and faithfulness are computed, not judged**, for every answer (`metrics` in the response, and Phase 12's report).
- **`PROMPT_VERSION` goes into every evaluation report.** A prompt change is a new baseline.
- **Open:**
  - claims that paraphrase without any quote can only be `partial`;
  - multi-turn follow-ups are answered from their own retrieval (history is shown, not fed back);
  - the 4B model sometimes refuses when the context *does* contain a two-hop answer (Phase 12 counts these as generation failures).
