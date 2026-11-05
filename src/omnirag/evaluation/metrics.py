"""Retrieval metrics on a ranked list of retrieved units (chunks the LLM would read).

Relevance is binary: a unit is relevant if it covers at least one fact of the question.

* **Hit@K**: at least one fact covered in the top K.
* **Recall@K** (= context recall here): share of the question's facts covered by the top K.
  A two-fact multi-hop question needs both facts for 1.0.
* **Precision@K**: share of the top-K units that are relevant.
* **MRR**: 1 / rank of the first relevant unit (0 if none in the top K).
* **nDCG@K**: binary gains; the ideal ranking puts all relevant units of the index first.
* **Context precision** (token-weighted): share of the top-K *tokens* in relevant units. It
  penalises big chunks that contain the answer plus a lot of noise.

The RAGAS-style, LLM-judged context precision/recall come with Phase 12. These are their
retrieval-only counterparts, computed from text containment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    unit_id: str
    tokens: int
    facts: frozenset[int]  # which of the question's facts this unit covers


def ranking_metrics(ranked: list[Unit], n_facts: int, n_relevant_in_index: int, k: int) -> dict:
    top = ranked[:k]
    relevant = [bool(u.facts) for u in top]
    covered = set().union(*(u.facts for u in top)) if top else set()
    first = next((i for i, r in enumerate(relevant) if r), None)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(relevant) if r)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, n_relevant_in_index)))
    tokens = sum(u.tokens for u in top)
    return {
        "hit": float(bool(covered)),
        "recall": len(covered) / n_facts if n_facts else 0.0,
        "precision": sum(relevant) / len(top) if top else 0.0,
        "mrr": 1 / (first + 1) if first is not None else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
        "context_precision": sum(u.tokens for u in top if u.facts) / tokens if tokens else 0.0,
    }
