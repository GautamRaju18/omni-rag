"""Router accuracy: rule router vs LLM router against labelled question intents (Phase 9b).

    uv run python -m omnirag.evaluation.routing_eval [--out x.json]

Labels: the seed and relational sets' question ``type`` mapped to an intent (``LABELS``),
and ``data/eval/routing_holdout.json`` (``intent``, labelled before the rules were written).
The seed set's "table" type says where the evidence *is* (a table), which the question
often doesn't show ("When is milestone M3 due?"): misses there are expected for any router
that only reads the question.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.llm.chat import OllamaChat
from omnirag.retrieval.router import INTENTS, LLMRouter, RuleRouter

EVAL = Path(__file__).resolve().parents[3] / "data" / "eval"
SETS = ("chunking_seed.json", "relational_seed.json", "routing_holdout.json")
LABELS = {"factual": "lookup", "keyword": "lookup", "unanswerable": "lookup", "table": "table",
          "image": "image", "multi_hop": "multi_hop", "comparative": "multi_hop",
          "relational": "multi_hop"}  # fmt: skip


def load(path: Path) -> list[tuple[str, str, str]]:
    """(id, question, intent) per question."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(q["id"], q["question"], q.get("intent") or LABELS[q["type"]])
            for q in data["questions"]]  # fmt: skip


def score(router, questions: list[tuple[str, str, str]]) -> dict:
    rows = [(qid, want, router.route(text)) for qid, text, want in questions]
    hits = sum(r.intent == want for _, want, r in rows)
    per_intent = {
        i: {"recall": round(sum(r.intent == i for _, w, r in rows if w == i)
                            / max(1, sum(w == i for _, w, _ in rows)), 3),
            "precision": round(sum(w == i for _, w, r in rows if r.intent == i)
                               / max(1, sum(r.intent == i for _, _, r in rows)), 3)}
        for i in INTENTS
    }  # fmt: skip
    return {
        "accuracy": round(hits / len(rows), 3),
        "correct": hits,
        "n": len(rows),
        "per_intent": per_intent,
        "confusion": {f"{w}->{g}": n for (w, g), n in
                      sorted(Counter((w, r.intent) for _, w, r in rows).items())},
        "errors": {qid: f"{w}->{r.intent}" for qid, w, r in rows if r.intent != w},
        "failed": sum(r.method == "llm_error" for _, _, r in rows),
        "p50_ms": round(statistics.median(r.ms for _, _, r in rows), 3),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.evaluation.routing_eval")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    settings = get_settings()
    llm = OllamaChat(settings.ollama_base_url, settings.generation_model,
                     timeout_s=settings.workflow_llm_timeout_s)  # fmt: skip
    llm.chat("Reply with OK.", "warm up", num_predict=2)
    routers = {"rules": RuleRouter(), "llm": LLMRouter(llm)}
    results: dict[str, dict] = {}
    for name in SETS:
        questions = load(EVAL / name)
        results[name] = {r: score(router, questions) for r, router in routers.items()}
        for r, out in results[name].items():
            print(f"{name:<22} {r:<6} {out['correct']:>2}/{out['n']} = {out['accuracy']:.3f}  "
                  f"p50 {out['p50_ms']} ms  errors {out['errors']}")  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps({
            "created": datetime.now(UTC).isoformat(), "model": settings.generation_model,
            "results": results}, indent=2, ensure_ascii=False) + "\n")  # fmt: skip
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
