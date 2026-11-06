# OmniRAG: portfolio and resume notes

Material for a resume, a portfolio page and interviews. Every number below is measured and links to its source file; re-run the commands to reproduce them.

## One-line summary
Local-first multimodal GraphRAG platform: ingests PDFs, scans, Office files, images and HTML into a Neo4j knowledge graph (vectors + full-text + validated entities). It answers questions with a local 4B model through a bounded LangGraph workflow, and checks every citation against the source text. A 110-question evaluation harness measures every stage.

## Resume bullets (pick 3-4)
- Built **OmniRAG**, a local-first multimodal GraphRAG platform (FastAPI, Neo4j 2026, LangGraph, Ollama, Streamlit, Docker) running on a 6 GB laptop GPU. It ingests 10 document types, including OCR for scans and vision-model image descriptions, with element-level provenance.
- Designed **claim-level citation verification**: the model must return a verbatim quote per claim, and code checks quotes and numbers against the cited passage. It corrects wrong source numbers and removes unsupported claims, so faithfulness and citation accuracy are computed for every answer, not judged.
- Implemented **hybrid retrieval** (bge-m3 dense + identifier-aware BM25 + graph traversal, Reciprocal Rank Fusion, cross-encoder reranking) and a **bounded agentic workflow** (rule router, grounded search agent, context check, corrective retry). Each component was kept or rejected by measured per-question wins and losses.
- Built a **110-question evaluation framework** (8 question types, 10 unanswerable, verbatim evidence verified by tests). It measures retrieval (Recall/MRR/nDCG/context recall), generation (correctness, faithfulness, citation accuracy, hallucination, refusals, LLM judge) and performance across 7 configurations. Every failure is attributed to the pipeline stage that lost it.
- Shipped the system as **one Docker image** (API + dashboard; Ollama on the host GPU) with 419 unit and 37 live integration tests, 21 architecture decision records and a learning guide per phase.

## Interview talking points
1. **"Why not trust the model's citations?"** A "[2]" is two characters it can type. Asking for a quote turns a citation into something code can search. Two rules came from real failures:
   - a quote that drops a table column counts only on one line;
   - a quote found only in a generated image description makes the claim "inferred".
2. **"What surprised you in the measurements?"**
   - Query rewriting, HyDE and graph retrieval did *not* help on this corpus (Phases 7 and 9a), so they are opt-in.
   - A token budget beat a fixed top-k (Phase 9c).
   - A 4B LLM router lost to ten regular expressions on three label sets (Phase 9b).
   - Expansion and lexical compression lost evidence (Phase 9c).
3. **"How do you know a failure's cause?"** `evaluation/failures.py` follows each fact through the pipeline (parsing → chunking → graph → embedding → retrieval → reranking → context → generation → citation) and blames the first stage that lost it.
4. **Hardware constraints as design input.**
   - 6 GB of VRAM holds the 4B generator and the embedder, nothing else.
   - The reranker runs on CPU (10–17 s), so it is a quality mode, not the default.
   - The judge is the same local model, so its agreement with deterministic checks is reported.
   - The laptop GPU throttles (measured 88 °C, 307 MHz), so every report records the GPU state.
5. **Safety and privacy.** Everything runs locally; LangSmith is opt-in and redacting; Cypher is always parameterized; user text never reaches Lucene unescaped; uploaded files are validated and served as attachments with `nosniff`.

## Numbers to quote (2026-11-06, all measured; [ADR-0020](adr/0020-evaluation-framework.md))
- 110-question evaluation, dense + BM25: Recall@5 **0.950**, context recall **0.980**, answer correctness **0.900**, quote-verified faithfulness **0.853**, **9/10** unanswerable questions refused.
- Cross-encoder reranking on the 76 questions it covered: answer correctness **0.932** (vs 0.918 for dense + BM25 on the same questions), MRR **0.966**, at 4.6× the latency on CPU.
- BM25 alone vs hybrid: cross-document correctness 0.571 vs 0.857; 14 of BM25's 15 retrieval failures are two-fact questions (multi-hop, cross-document, comparative) where it finds one fact and misses the other.
- A 4B LLM judge called 25 of 27 refusals "correct": refusals are scored by rule, and the judge's agreement with deterministic checks (0.95–0.96) is reported.
- Agentic retrieval (Phase 9b): relational recall 0.875 → 0.938, MRR 0.838 → 0.917. The rule router beats the LLM router (30/40 vs 23/40).
- Token-budget context (Phase 9c): context recall 0.986 → 1.000 (seed) and 0.875 → 0.938 (relational) over a fixed top 5.

## What I would do next
- A larger, independently written question set: the current one has a single author, so it can't rule out unconscious bias towards the system's strengths.
- GPU reranking (needs a CUDA torch decision, ADR-0015).
- Multi-turn query rewriting from chat history.
- PostgreSQL for a multi-process API.
