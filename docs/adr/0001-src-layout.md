# ADR-0001: Single `src/omnirag` package instead of top-level module folders

- Status: Accepted (2026-11-04)

## Context
The original brief sketched top-level folders: `backend/`, `ingestion/`, `retrieval/`, `evaluation/`, and so on. Each would become its own top-level import name.

## Decision
Use one installable package, `src/omnirag/`, with subpackages that map one-to-one onto the brief's modules: `api/` (was `backend/`), `core/`, `schemas/`, `llm/` (generation client), `knowledge_base/`, and later `ingestion/`, `chunking/`, `embeddings/`, `retrieval/`, `generation/`, `workflows/`, `evaluation/`. `frontend/`, `tests/`, `data/` and `docs/` stay at the repo root. A phase's subpackages are created when that phase starts, not before.

## Consequences
- Generic names like `evaluation` or `retrieval` can't shadow PyPI packages or each other.
- The src layout means tests import the **installed** package, so a missing `__init__.py` or packaging error shows up in CI instead of hiding behind the CWD on `sys.path`.
- Imports are explicit (`from omnirag.retrieval.fusion import rrf`), which helps readers and tooling.
- Slightly longer import paths. That's acceptable.
