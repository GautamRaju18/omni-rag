"""Analytics over stored chat traces and evaluation reports (Phase 13, ``GET /analytics``).

Everything is computed from records, never estimated:

* **answer quality** (live, no ground truth): answered / refused (and why), grounded answers,
  claim verification status, citation accuracy, flagged failure kinds (``failures.live_flags``);
* **system performance**: latency p50 / p95 of every component in the stored traces;
* **retrieval**: verdicts of the context check, route intents, context tokens used;
* **evaluation**: the newest completed report's per-experiment summary and failure root causes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from omnirag.evaluation.failures import live_flags


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, max(0, round(p * len(v)) - 1))], 1)


def turn_analytics(turns: list[dict[str, Any]]) -> dict[str, Any]:
    answers = [t["answer"] for t in turns]
    answered = [a for a in answers if a.get("answerable")]
    statuses: Counter = Counter()
    for a in answers:
        for c in a.get("claims", []):
            statuses[c.get("status")] += 1
    accuracy = [a["metrics"]["citation_accuracy"] for a in answered
                if a.get("metrics", {}).get("citation_accuracy") is not None]  # fmt: skip
    components: dict[str, list[float]] = {}
    for t in turns:
        for key, ms in (t.get("timings") or {}).items():
            if isinstance(ms, int | float):
                components.setdefault(key, []).append(ms)
    flags = Counter(f for t in turns for f in live_flags(t))
    retrieval = [t.get("retrieval") or {} for t in turns]
    return {
        "turns": len(turns),
        "conversations": len({t.get("conversation_id") for t in turns}),
        "answered": len(answered),
        "grounded": sum(1 for a in answers if a.get("grounded")),
        "refused_by": dict(Counter(a.get("refused_by") for a in answers if a.get("refused_by"))),
        "claim_status": dict(statuses),
        "citation_accuracy_mean": round(sum(accuracy) / len(accuracy), 3) if accuracy else None,
        "failure_flags": dict(flags.most_common()),
        "verdicts": dict(Counter(r.get("verdict") for r in retrieval if r.get("verdict"))),
        "routes": dict(Counter((r.get("route") or {}).get("intent") for r in retrieval
                               if r.get("route"))),
        "context_tokens_p50": _pct([(r.get("context") or {}).get("tokens", 0) for r in retrieval
                                    if r.get("context")], 0.5),
        "latency_ms": {k: {"p50": _pct(v, 0.5), "p95": _pct(v, 0.95), "n": len(v)}
                       for k, v in sorted(components.items())},
        "recent": [{"turn_id": t["id"], "conversation_id": t.get("conversation_id"),
                    "created_at": t.get("created_at"), "question": t["question"],
                    "answered": t["answer"].get("answerable"),
                    "grounded": t["answer"].get("grounded"),
                    "refused_by": t["answer"].get("refused_by"),
                    "total_ms": (t.get("timings") or {}).get("total_ms"),
                    "flags": live_flags(t)} for t in turns[:20]],
    }  # fmt: skip


def evaluation_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    return {
        "created": report.get("created"), "dataset": report.get("dataset"),
        "questions": report.get("questions"), "generation_model": report.get("generation_model"),
        "experiments": {name: {"label": e["label"], "retrieval": e["retrieval"],
                               "generation": e["generation"], "judge": e["judge"],
                               "performance": e["performance"],
                               "root_causes": e["failures"]["root_causes"]}
                        for name, e in report.get("experiments", {}).items()},
    }  # fmt: skip
