# Phase 11: Citation and answer validation: the model proposes, code checks

> Decision record: [ADR-0019](../adr/0019-grounded-generation-and-citations.md). Code: `src/omnirag/generation/citations.py`. Tests: `tests/unit/test_generation.py`.

## 1. Problem being solved
"[2]" after a sentence is two characters a model can type. A citation is only worth something if it can be **verified**:
- do the cited words exist in the cited source?
- do the numbers in the claim match the source?
- is the claim resting on the document's own text, or on a description a vision model wrote?

This phase turns every answer into checked claims, so faithfulness and citation accuracy are *computed* for every answer, not estimated.

## 2. Concepts
| Concept | Meaning here |
|---|---|
| **Verified** | The normalised quote is in a cited block (and every number of the claim is too). |
| **Relocated** | The quote is real but in a block the model didn't cite: the citation is corrected and flagged. Never invented. |
| **Elided quote** | The model left out columns of one table row (`Category: 300 \| AUR-C106: 96.2`). Accepted only if the parts appear in order on **one** line. |
| **Partial** | No usable quote, but the claim's numbers, identifiers and most content words are in the cited blocks. Shown with a ⚠. |
| **Unsupported** | Nothing checks out: removed from the answer, kept in `claims` (hallucination detection). |
| **Generated evidence** | The quote exists only inside an "Image description (generated …)" line: the claim becomes `inferred`. |
| **Citation accuracy** | Share of claims whose *model-cited* source contains the quote. |
| **Faithfulness** | Share of claims that are verified or relocated. |

## 3. How a claim is checked (in order)
1. Drop source numbers outside `1..len(blocks)` and record them in `issues`.
2. Normalise the quote (`knowledge_base.entities.norm`: NFKC, so NaPF₆ = NaPF6; unified dashes; no markdown; case; whitespace).
3. `match(quote, block)` for every block: `verbatim`, `elided` (one line, parts in order), or nothing.
4. Found in a cited block → `verified`; only elsewhere → `relocated`, with the citation moved there.
5. **Number check.** Every number in the claim text (thousands separators removed; digits inside identifiers like AUR-C103 ignored) must occur in the claim's blocks. Otherwise `partial` with an issue: a true quote can hide a changed number.
6. No quote match:
   - `partial` if numbers, identifiers and ≥ 50% of the content words occur;
   - otherwise `unsupported`.
7. If the quote is not in the block's *original* lines (generated lines removed), it rests on model-written text: `generated_evidence`, and the claim is `inferred`.
8. `direct` survives only for `verified` or `relocated`.

## 4. Two rules found by running it
- **Elided quotes.** Live run, "AUR-C106 after 300 cycles at 45 °C": the model quoted `Category: 300 | AUR-C106: 96.2`. The real line is `Category: 300 | AUR-C105: 95.9 | AUR-C106: 96.2`. Rejecting it would call a correct, checkable citation "not verbatim"; accepting fragments from anywhere would let `300` from one line and `94.1` from another "verify" a wrong value. Hence: parts in order, same line (`test_elided_table_quote_counts_only_within_one_line` has both cases).
- **OCR versus description.** "Pilot batch: 100 cells (M3)" is in the whiteboard's OCR text *and* in the vision model's description. The first version flagged it as generated evidence because the description contained it. The rule is now "only in generated text" (`test_text_also_in_ocr_is_original_evidence`).

## 5. How to run / test
```bash
uv run pytest tests/unit/test_generation.py -q        # 16 tests: every status, numbers, streaming, store, API
uv run python -m omnirag.evaluation.rag_eval --experiments E7 --limit 20   # faithfulness + citation accuracy on real answers
```
In the dashboard's **Ask** page, open "How this answer was checked": every claim, its status (icon + word), what the model cited vs what code kept, the quote, and the issues.

## 6. Expected outputs
- `test_validate_metrics_citations_and_rendered_answer`: claims `[verified, relocated, unsupported]` → citations [1, 2], citation accuracy 0.333, faithfulness 0.667, and the rendered answer leaves out the unsupported claim.
- Live (Phase 10 §7): "Why did cell AUR-C103 fail?" has 2/2 verified; the 8-claim comparison has 8/8 verified.
- Measured on 110 questions × 7 configurations: [phase-12](phase-12-evaluation.md).

## 7. Common errors
| Symptom | Cause / fix |
|---|---|
| Many `partial` claims with "quote not found verbatim" | The model paraphrased the quote. Check the prompt (rule 2) and `PROMPT_VERSION`; quotes > 40 words get trimmed by the model. |
| `relocated` claims | The model cited the wrong block number. They are corrected; Phase 12 counts them as citation failures. |
| "numbers not in the cited sources: [15.0]" | The paraphrase changed a number: treat as a hallucination even if the quote is real. |
| A table-row claim is `partial` | The quote mixed two rows: by design not accepted. |

## 8. Exercise (≈30 min)
Write three claims by hand for "What is the planned budget of the electrolyte workstream?" against the budget block:
- one verified;
- one with a changed number;
- one quoting the status report instead.

Run `validate` on them and predict each status and `citation_accuracy` before you look.

## 9. Acceptance criteria: Phase 11
- [ ] `uv run pytest tests/unit/test_generation.py` passes (every status, elided quotes, OCR vs generated, numbers).
- [ ] On a live answer, every shown citation's text contains its claim's quote (open the citation cards on the Ask page).
- [ ] Phase 12's report lists faithfulness, citation accuracy and hallucination rate per configuration.
