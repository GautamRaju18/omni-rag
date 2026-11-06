# Phase 10: Grounded generation with Ollama: prompts, structured output, streaming, chat history

> Decision record: [ADR-0019](../adr/0019-grounded-generation-and-citations.md). Citation checking (Phase 11) is in [phase-11](phase-11-citations-and-validation.md); the 110-question measurement in [phase-12](phase-12-evaluation.md).

## 1. Problem being solved
Phases 1–9 find evidence. Now a local 4B model has to **answer** from it, and the brief adds four hard requirements:
- no unsupported claims;
- document names and pages for every claim;
- "insufficient evidence" when the documents don't say;
- direct evidence kept apart from inference.

The difficulty: a language model produces plausible text whatever its context contains. So the design question is **what must the model produce so that code can check it?**

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Grounding** | Every statement can be traced to words in a retrieved passage. |
| **Structured output** | Ollama's `format` = a JSON schema: the reply always parses and enums are respected. The *content* is still untrusted. |
| **Claim** | One fact of the answer, with the source numbers, a verbatim quote, and `direct`/`inferred` support. |
| **Prompt template / version** | The system prompt + user template + schema, versioned (`answer-v1`); evaluation reports record it. |
| **Refusal** | An answer that says the documents don't contain it. Here it can come from the context check (no model call) or from the model. |
| **Streaming (SSE)** | Server-Sent Events: the server writes `event:`/`data:` lines while it works; the client renders as they arrive. |
| **Token budget** | 1,024 bge-m3 tokens of context (~1,070 qwen3 tokens, Phase 9c) + the system prompt, inside `num_ctx = 8192`. |

## 3. Architecture
```
POST /chat ─► ChatService.ask
               ├─ RetrievalWorkflow.retrieve (Phase 9: route, agent, check, assemble)
               ├─ AnswerGenerator.generate
               │    ├─ refuse without a call: empty context, or check = insufficient
               │    ├─ one OllamaChat.chat(ANSWER_SYSTEM, sources + question, ANSWER_SCHEMA)
               │    └─ citations.validate → Answer (claims, citations, metrics)
               └─ ChatStore.add_turn (SQLite: the whole trace)
POST /chat/stream ─► same, OllamaChat.stream → ClaimStream → SSE: start, retrieval, claim×n, answer, done
GET /chat/history[/{id}], DELETE /chat/history/{id}, GET /chat/turns/{id}
```
Files:
```
src/omnirag/llm/chat.py            + stream(): token deltas, retries before the first token only
src/omnirag/generation/prompts.py   ANSWER_SYSTEM, ANSWER_SCHEMA, user_prompt, PROMPT_VERSION
src/omnirag/generation/models.py    Claim, Citation, AnswerMetrics, Answer
src/omnirag/generation/generator.py AnswerGenerator (generate, stream), ClaimStream
src/omnirag/generation/citations.py validate_claim, validate, render_answer   (Phase 11)
src/omnirag/generation/history.py   ChatStore (SQLite, WAL, connection per call)
src/omnirag/generation/chat.py      ChatService (ask, stream), build_chat, timings
src/omnirag/api/routers/chat.py     the endpoints
```

## 4. Technology choices and alternatives
- **One structured call, not "answer then cite".** A two-step design (free-text answer, then a second call to attach citations) doubles latency and lets the citer rationalise. Asking for claims with quotes in one pass gives code something to check.
- **Quotes instead of sentence-ID pointers.** Numbering every sentence of the context would make citations exact, but it bloats the prompt and the model still has to say *which* words. A quote can be searched in the text.
- **Refusing before the call.** A grader-only design (always generate, then judge) spends 3–15 s to produce an answer that would be thrown away, and risks a confident hallucination. The identifier check is deterministic and free.
- **SQLite for history.** Postgres would add a container to an 8 GB VM for kilobytes of data in a one-process API (ADR-0019 explains the swap path).
- **SSE over WebSockets.** The flow is one-way (server → client), and SSE is plain HTTP that FastAPI's `StreamingResponse` serves.

## 5. Code walkthrough
- `prompts.py`: the system prompt's six rules map one-to-one onto the brief's seven requirements (the module docstring lists the mapping). Sources come first and the question last, so the model reads the evidence before the task. The schema orders `answerable` → `claims` → `missing`.
- `generator.py`:
  - `_refusal` handles empty context and an `insufficient` check before any call;
  - `_finish` validates the claims and sets `answerable` only when at least one claim survived;
  - `stream()` feeds every token delta to `ClaimStream`, a small JSON scanner (string and escape aware) that yields each complete object inside the `claims` array, and validates it at once.
- `chat.py`: `retrieval_summary` keeps the trace without the bulky rendered text; `timings` turns the workflow trace into per-component milliseconds (route, retrieve, plan, check, assemble, embedding, generation, total).
- `history.py`: two tables (`conversations`, `turns`), cascade delete, WAL mode so the dashboard can read during a write, and one connection per call (safe from worker threads).

## 6. How to run
```bash
cd /mnt/p/omni-rag && git pull && uv sync
uv run uvicorn omnirag.api.main:app --no-access-log
curl -s localhost:8000/api/v1/chat -H 'content-type: application/json' \
     -d '{"question": "Why did cell AUR-C103 fail?"}' | jq '.answer | {text, grounded, metrics}'
curl -N localhost:8000/api/v1/chat/stream -H 'content-type: application/json' \
     -d '{"question": "Who prepared the safety review of the E-5 failure?"}'
curl -s localhost:8000/api/v1/chat/history | jq
```
Dashboard: **Ask** page (`uv run streamlit run src/omnirag/frontend/app.py`).

## 7. Expected outputs (measured 2026-11-06, qwen3:4b-instruct, defaults)
```
=== Why did cell AUR-C103 fail?
  answerable=True grounded=True verdict=sufficient metrics={claims: 2, verified: 2, unsupported: 0, citation_accuracy: 1.0, faithfulness: 1.0}
  ANSWER: Cell AUR-C103 failed due to gas evolution at 45 °C, causing a 14% pouch swelling [1]. The failure of cell
  AUR-C103 was caused by electrolyte decomposition of the E-5 baseline, not a manufacturing defect [4].
  cites: [1] aurora_research_notes.md; [4] meeting_notes_2026-03-12.txt
  timings: retrieve 440 · check 1,340 · assemble 812 (first call loads the tokenizer) · generation 3,615 · total 6,214 ms
  tokens in/out 1319/196

=== What capacity retention did cell AUR-C109 reach?
  answerable=False refused=check  missing: No source mentions AUR-C109.   total 26 ms, 0 model calls

=== Which electrolyte performs better at 45 °C, E-5 + FEC or E-7?
  8 claims, all verified, from aurora_safety_review.pdf p2, cell_test_results.csv, e7_technical_note.pdf p1
  generation 15.3 s (718 output tokens)
```
The full 110-question numbers per retrieval configuration are in Phase 12.

## 8. Common errors
| Symptom | Cause / fix |
|---|---|
| `refused_by: "error"`, `error: "...timed out"` | Ollama busy or the GPU throttled; raise `OMNIRAG_GENERATION_TIMEOUT_S` (default 180). |
| `refused_by: "check"` on a question you think is answerable | An identifier in the question is in no passage (a typo like `AUR-C1O3`?). `respect_check: false` generates anyway. |
| 404 on `POST /chat` with `conversation_id` | The conversation was deleted, or the ID is from another data directory. |
| The stream shows claims but the final answer has fewer | Unsupported claims are removed from `text` (still listed in `claims`). |
| Answers are slow (5–15 s) | Laptop GPU thermal throttling: check `nvidia-smi` clock vs max. The evaluation report records it. |

## 9. Exercise (≈45 min)
1. Ask the dashboard "Which electrolyte performs better at 45 °C?" and open "How this answer was checked". Which claims are `direct`, which `inferred`? Do you agree?
2. Change rule 6 of `ANSWER_SYSTEM` (copy numbers exactly) to something weaker, bump `PROMPT_VERSION`, and run `rag_eval --experiments E7 --limit 20`. What happens to faithfulness? Revert.
3. **Think.** Why must the refusal for AUR-C109 happen *before* the model call rather than by asking the model "is this answerable"?

## 10. Acceptance criteria: Phase 10
- [ ] `POST /chat` on "Why did cell AUR-C103 fail?" returns `grounded: true`, at least one `verified` claim, and citations with chunk IDs.
- [ ] "What capacity retention did cell AUR-C109 reach?" is refused by the check in well under a second.
- [ ] `POST /chat/stream` emits `start`, `retrieval`, at least one `claim`, `answer`, `done`.
- [ ] `GET /chat/history` lists the conversation; `DELETE` removes it and its turns.
- [ ] `uv run pytest` and `uv run pytest -m integration` pass.
