"""Pure helpers for the dashboard: API JSON in, tables / Graphviz DOT out (no Streamlit here,
so they're unit-tested without a browser)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Workflow nodes in reading order, and what each one does (shown next to the diagram).
NODES: dict[str, str] = {
    "route": "Classifies the question (lookup / table / image / multi-hop) and, with "
             "self-query, finds documents it names. Both become extra *preference* lists.",
    "transform": "Optionally rewrites the question (rewrite, multi-query, HyDE). The original "
                 "question always stays.",
    "retrieve": "Runs dense (vector) and BM25 (keyword) search for every query, in parallel, "
                "and fuses *all* lists so far with Reciprocal Rank Fusion.",
    "plan": "The agent reads the current top passages and picks the next search, a graph "
            "lookup, or finish. Its queries may only use names it has read.",
    "rerank": "A cross-encoder re-scores the fused candidates (slow on CPU).",
    "check": "Is the context enough? First every identifier of the question must appear, "
             "then the LLM grader says sufficient / partial / insufficient.",
    "correct": "Writes one new query aimed at what the check says is missing.",
}  # fmt: skip
EDGES = [("START", "route"), ("route", "transform"), ("transform", "retrieve"),
         ("retrieve", "plan"), ("plan", "retrieve"), ("retrieve", "rerank"),
         ("retrieve", "check"), ("plan", "rerank"), ("plan", "check"), ("rerank", "check"),
         ("check", "correct"), ("correct", "retrieve"), ("check", "END")]  # fmt: skip
VARIANTS = ("documents", "table", "image", "lookup", "multi_hop")
# Reference palette (dataviz skill): slot 1 blue marks what ran; muted gray what didn't.
RAN, IDLE, INK_MUTED = "#2a78d6", "#a3a29c", "#7a7974"


def pipeline_dot(trace: list[dict[str, Any]]) -> str:
    """The workflow graph with the path this run took: nodes that ran are filled and show
    how often and how long; edges taken are solid."""
    order = ["START", *[t["node"] for t in trace], "END"]
    taken = {(a, b) for a, b in zip(order, order[1:], strict=False)}
    runs: dict[str, list[float]] = {}
    for t in trace:
        runs.setdefault(t["node"], []).append(float(t.get("ms", 0.0)))
    lines = ['digraph G {', 'rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;',
             f'node [shape=box style="rounded" fontname="Helvetica" fontsize=11 '
             f'color="{IDLE}" fontcolor="{INK_MUTED}"];',
             f'edge [color="{IDLE}" arrowsize=0.6 fontname="Helvetica" fontsize=9];',
             f'START [shape=circle label="" width=0.15 style=filled color="{RAN}"];',
             f'END [shape=doublecircle label="" width=0.12 style=filled color="{RAN}"];',
             ]  # fmt: skip
    for node in NODES:
        if node in runs:
            ms = sum(runs[node])
            label = f"{node}\\n×{len(runs[node])} · {ms:,.0f} ms"
            lines.append(f'{node} [label="{label}" style="rounded,filled" fillcolor="{RAN}" '
                         f'color="{RAN}" fontcolor="white"];')  # fmt: skip
        else:
            lines.append(f'{node} [label="{node}" style="rounded,dashed"];')
    for a, b in EDGES:
        style = f'color="{RAN}" penwidth=2' if (a, b) in taken else 'style="dashed"'
        lines.append(f"{a} -> {b} [{style}];")
    lines.append("}")
    return "\n".join(lines)


def node_summary(t: dict[str, Any]) -> str:
    """One line on what a node did in this run (from its trace entry)."""
    n = t["node"]
    if n == "route":
        extra = [f"prefers {', '.join(t['prefer'])}"] if t.get("prefer") else []
        extra += [f"documents {t['documents']}"] if t.get("documents") else []
        extra += ["agent on"] if t.get("agent") else []
        return f"intent {t.get('intent')} via {t.get('method')}" + (
            f" · {'; '.join(extra)}" if extra else ""
        )
    if n == "transform":
        out = f"{t.get('transform')}: +{len(t.get('queries', []))} queries"
        out += " · HyDE passage" if t.get("hyde") else ""
        return out + (f" · errors {t['errors']}" if t.get("errors") else "")
    if n == "retrieve":
        return (
            f"{t.get('new_lists')} new lists ({t.get('lists')} in total) → "
            f"{t.get('candidates')} fused candidates"
        )
    if n == "plan":
        s = f"{t.get('action')} {t.get('query') or ''}".strip()
        return s + (f" · REFUSED: {t['refused']}" if t.get("refused") else "")
    if n == "rerank":
        return f"{t.get('pairs')} (question, passage) pairs scored"
    if n == "check":
        return f"{t.get('verdict')} via {t.get('method')}"
    if n == "correct":
        return f"new query: {t.get('query')}" + (f" · error {t['error']}" if t.get("error")
                                                 else "")  # fmt: skip
    return ""


def timeline_rows(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows, start = [], 0.0
    for i, t in enumerate(trace, start=1):
        ms = float(t.get("ms", 0.0))
        rows.append({"step": i, "node": t["node"], "label": f"{i}. {t['node']}",
                     "start_ms": round(start, 1), "end_ms": round(start + ms, 1),
                     "ms": round(ms, 1), "what happened": node_summary(t)})  # fmt: skip
        start += ms
    return rows


def parse_list_key(key: str) -> dict[str, str]:
    """``dense:Q`` / ``bm25:image:Q`` / ``dense:hyde`` / ``graph:E-7`` → parts."""
    retriever, _, rest = key.partition(":")
    if retriever == "dense" and rest == "hyde":
        return {"retriever": "dense", "variant": "hyde", "query": "(HyDE passage)"}
    head, sep, tail = rest.partition(":")
    if sep and head in VARIANTS:
        return {"retriever": retriever, "variant": head, "query": tail}
    variant = "entity" if retriever == "graph" else ""
    return {"retriever": retriever, "variant": variant, "query": rest}


def list_rows(lists: dict[str, Any], question: str) -> list[dict[str, Any]]:
    rows = []
    for key, info in lists.items():
        p = parse_list_key(key)
        hits = info.get("hits", [])
        rows.append({
            "list": key, "retriever": p["retriever"],
            "preference": p["variant"] or "—",
            "query": "(the question)" if p["query"] == question else p["query"],
            "hits": len(hits),
            "top 3": ", ".join(h[3] if len(h) > 3 else h[0] for h in hits[:3]),
            "ms": info.get("latency_ms"),
        })  # fmt: skip
    return rows


def contribution_rows(hits: list[dict[str, Any]], rrf_k: int = 60) -> list[dict[str, Any]]:
    """Why each final chunk ranks where it does: its rank in every list and the RRF share
    1/(k + rank) that list added. Reranked hits show the cross-encoder score instead."""
    rows = []
    for h in hits:
        sources = h.get("sources", {})
        parts = {name: s for name, s in sources.items() if name != "rerank"}
        row = {"rank": h["rank"], "file": h["source_file"], "type": h["element_type"],
               "rrf": round(sum(1 / (rrf_k + s["rank"]) for s in parts.values()), 4),
               "in lists": len(parts),
               "ranks (best first)": " · ".join(
                   f"{_short(n)} #{int(s['rank'])}"
                   for n, s in sorted(parts.items(), key=lambda x: x[1]["rank"])),
               "chunk": h["chunk_id"]}  # fmt: skip
        rows.append(row)
    return rows


def _short(list_key: str) -> str:
    p = parse_list_key(list_key)
    q = p["query"] if len(p["query"]) <= 28 else p["query"][:25] + "…"
    return f"{p['retriever']}{'/' + p['variant'] if p['variant'] else ''}:{q}"


def compare_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per chunk found by any mode, with its rank in each mode (None = not found)."""
    seen: dict[str, dict[str, Any]] = {}
    for mode, result in results.items():
        for h in result.get("hits", []):
            row = seen.setdefault(h["chunk_id"], {
                "chunk": h["chunk_id"], "file": h["source_file"], "type": h["element_type"],
                "text": " ".join(h["content"].split())[:90]})  # fmt: skip
            row[mode] = h["rank"]
    modes = list(results)
    return sorted(seen.values(), key=lambda r: (min(r.get(m) or 99 for m in modes), r["chunk"]))


def overlap(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """How many chunks two modes' top-k share."""
    ids = {m: {h["chunk_id"] for h in r.get("hits", [])} for m, r in results.items()}
    return [{"mode": a, **{b: len(ids[a] & ids[b]) for b in ids}} for a in ids]


METRICS = ("recall", "mrr", "ndcg", "hit", "precision")


def benchmark_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.json") if p.is_file())


def benchmark_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of a benchmark JSON: one per configuration that reports retrieval metrics."""
    rows = []
    for name, r in (data.get("results") or {}).items():
        if not isinstance(r, dict) or "recall" not in r:
            continue
        lat = r.get("latency_ms") or {}
        rows.append({"config": name, **{m: r.get(m) for m in METRICS if m in r},
                     "p50 ms": lat.get("p50") if isinstance(lat, dict) else None,
                     "llm calls/q": r.get("llm_calls_per_question")})  # fmt: skip
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def example_questions(eval_dir: Path) -> list[tuple[str, str]]:
    """(label, question) from the eval sets, for the question picker."""
    out = []
    for name in ("chunking_seed.json", "relational_seed.json"):
        path = eval_dir / name
        if path.exists():
            for q in load_json(path)["questions"]:
                out.append((f"{q['id']} · {q.get('type', '')} · {q['question']}", q["question"]))
    return out
