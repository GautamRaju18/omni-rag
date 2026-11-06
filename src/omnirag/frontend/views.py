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
    "assemble": "Builds the context the generator will read: blocks in rank order within a "
                "token budget, optionally expanded (neighbours / section) and compressed "
                "(kept sentences and table rows, verbatim).",
}  # fmt: skip
EDGES = [("START", "route"), ("route", "transform"), ("transform", "retrieve"),
         ("retrieve", "plan"), ("plan", "retrieve"), ("retrieve", "rerank"),
         ("retrieve", "check"), ("plan", "rerank"), ("plan", "check"), ("rerank", "check"),
         ("check", "correct"), ("correct", "retrieve"), ("check", "assemble"),
         ("assemble", "END")]  # fmt: skip
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
    if n == "assemble":
        if t.get("skipped"):
            return "no assembler configured"
        return (
            f"{t.get('blocks')} blocks · {t.get('tokens')}/{t.get('budget')} tokens · "
            f"expand {t.get('expand')} · compress {t.get('compress')}"
        )
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


def context_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per assembled block: where it comes from and what it costs."""
    return [{"#": i, "source": f"{b['source_file']}"
             + (f" > {' > '.join(b['section_path'])}" if b.get("section_path") else ""),
             "rank": b["rank"], "tokens": b["tokens"],
             "chunks": len(b["chunk_ids"]), "retrieved": len(b["hit_chunk_ids"]),
             "added by expansion": len(b["chunk_ids"]) - len(b["hit_chunk_ids"]),
             "units kept": (f"{b['units_kept']}/{b['units_total']}"
                            if b.get("units_total") is not None else "all")}
            for i, b in enumerate(context.get("blocks", []), start=1)]  # fmt: skip


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


METRICS = ("recall", "mrr", "ndcg", "hit", "precision", "context recall", "tokens p50")


def benchmark_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.json") if p.is_file())


def benchmark_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows of a benchmark JSON: one per configuration that reports retrieval metrics."""
    rows = []
    for name, r in (data.get("results") or {}).items():
        if isinstance(r, dict) and "context_recall" in r:  # context benchmark (Phase 9c)
            rows.append({"config": name, "context recall": r["context_recall"],
                         "tokens p50": r.get("tokens_p50"), "tokens max": r.get("tokens_max"),
                         "assembly ms p50": r.get("assembly_ms_p50"),
                         "llm calls/q": r.get("llm_calls_per_question")})  # fmt: skip
            continue
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
    for name in ("omnirag_eval_v1.json",):
        path = eval_dir / name
        if path.exists():
            for q in load_json(path)["questions"]:
                out.append((f"{q['id']} · {q.get('type', '')} · {q['question']}", q["question"]))
    return out


# ---------------------------------------------------------------- Phase 10-14 screens
# Status palette (dataviz reference, fixed): always shown with an icon and a word.
STATUS = {
    "verified": ("✅", "verified", "#0ca30c"),
    "relocated": ("↪️", "citation corrected", "#ec835a"),
    "partial": ("⚠️", "not verbatim", "#fab219"),
    "unsupported": ("⛔", "unsupported (removed)", "#d03b3b"),
}
# Categorical slots 1-3 in fixed order for the graph explorer's node kinds.
KIND_COLORS = {"document": "#2a78d6", "entity": "#eb6834", "chunk": "#1baf7a"}


def status_label(status: str) -> str:
    icon, word, _ = STATUS.get(status, ("•", status, IDLE))
    return f"{icon} {word}"


def claim_rows(answer: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"claim": c["text"], "check": status_label(c["status"]), "support": c["support"],
             "cites": " ".join(f"[{n}]" for n in c["sources"]) or "—",
             "model cited": " ".join(f"[{n}]" for n in c.get("model_sources", [])) or "—",
             "quote": c.get("quote", ""), "issues": "; ".join(c.get("issues", []))}
            for c in answer.get("claims", [])]  # fmt: skip


def citation_cards(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """Each cited source with its text (the block the model read) and the claims citing it."""
    blocks = ((turn.get("retrieval") or {}).get("context") or {}).get("blocks", [])
    answer = turn.get("answer", {})
    out = []
    for c in answer.get("citations", []):
        block = blocks[c["number"] - 1] if 0 < c["number"] <= len(blocks) else {}
        claims = [
            cl["text"]
            for cl in answer.get("claims", [])
            if c["number"] in cl["sources"] and cl["status"] != "unsupported"
        ]
        out.append(
            {
                **c,
                "text": block.get("text", ""),
                "claims": claims,
                "where": " > ".join(c.get("section_path") or []),
                "pages": ", ".join(str(p) for p in c.get("page_numbers") or []),
            }
        )
    return out


REFUSALS = {"check": "🛑 not in the documents (context check)",
            "model": "🛑 not in the documents (model)",
            "empty_context": "🛑 nothing retrieved"}  # fmt: skip


def answer_badge(answer: dict[str, Any]) -> str:
    if answer.get("refused_by") == "error":
        return "⛔ model error"
    if not answer.get("answerable"):
        return REFUSALS.get(answer.get("refused_by"), "🛑 refused")
    m = answer.get("metrics", {})
    if answer.get("grounded"):
        ok = m.get("verified", 0) + m.get("relocated", 0)
        return f"✅ grounded · {ok}/{m.get('claims')} claims quote-verified"
    return f"⚠️ {m.get('unsupported', 0)} unsupported claim(s) removed"


EVAL_COLUMNS = {
    "R@5": ("retrieval", "recall"), "MRR": ("retrieval", "mrr"), "nDCG@5": ("retrieval", "ndcg"),
    "context recall": ("retrieval", "context_recall"),
    "context precision": ("retrieval", "context_precision"),
    "answer correct": ("generation", "answer_correctness"),
    "completeness": ("generation", "completeness"),
    "faithfulness": ("generation", "faithfulness"),
    "citation accuracy": ("generation", "citation_accuracy"),
    "hallucination rate": ("generation", "hallucination_rate"),
    "judge correct": ("judge", "correct_rate"), "judge relevance": ("judge", "relevance"),
    "e2e p50 ms": ("performance", "end_to_end_ms_p50"),
    "e2e p95 ms": ("performance", "end_to_end_ms_p95"),
}  # fmt: skip


def eval_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per experiment: the comparison table (values from the report, unchanged)."""
    rows = []
    for name, e in (report.get("experiments") or {}).items():
        g = e.get("generation", {})
        n, total = e.get("questions"), report.get("questions")
        label = e.get("label", name)
        if e.get("complete") is False or (n and total and n < total):
            label += f" (partial: {n}/{total} questions)"
        row = {"exp": name, "configuration": label, "questions": n}
        row |= {col: (e.get(sec) or {}).get(key) for col, (sec, key) in EVAL_COLUMNS.items()}
        row["unanswerable refused"] = f"{g.get('unanswerable_refused')}/{e.get('unanswerable')}"
        row["false refusals"] = g.get("false_refusals")
        rows.append(row)
    return rows


def metric_rows(report: dict[str, Any], column: str) -> list[dict[str, Any]]:
    return [{"experiment": f"{r['exp']} · {r['configuration']}", "value": r[column]}
            for r in eval_rows(report) if r.get(column) is not None]  # fmt: skip


def failure_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Root-cause counts: one row per (experiment, category)."""
    return [{"experiment": name, "category": cat.replace("_", " "), "questions": n}
            for name, e in (report.get("experiments") or {}).items()
            for cat, n in (e.get("failures") or {}).get("root_causes", {}).items()]  # fmt: skip


def per_question_rows(report: dict[str, Any], experiment: str,
                      questions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:  # fmt: skip
    pq = ((report.get("experiments") or {}).get(experiment) or {}).get("per_question", {})
    return [{"id": qid, "type": questions.get(qid, {}).get("type"),
             "question": questions.get(qid, {}).get("question", ""),
             "correct": "✅" if r["correct"] else "❌", "refused": r["refused"],
             "hallucination": "⛔" if r["hallucination"] else "", "R@5": r.get("recall"),
             "context recall": r.get("context_recall"), "judge": r.get("judge"),
             "root cause": (r.get("failures") or ["—"])[0].replace("_", " ")}
            for qid, r in pq.items()]  # fmt: skip


def latency_rows(analytics: dict[str, Any]) -> list[dict[str, Any]]:
    lat = (analytics.get("chat") or {}).get("latency_ms", {})
    return [{"component": k.replace("_ms", "").replace("_", " "), "p50 ms": v["p50"],
             "p95 ms": v["p95"], "turns": v["n"]} for k, v in lat.items()]  # fmt: skip


def _dot_label(text: Any, width: int = 28) -> str:
    text = str(text)
    text = text if len(text) <= width else text[: width - 1] + "…"
    return text.replace("\\", "\\\\").replace('"', '\\"')


def graph_dot(g: dict[str, Any]) -> str:
    """Graphviz DOT for the explorer. Node IDs are generated (n0, n1, ...): never a DOT
    keyword like ``graph``, never user text; labels are escaped."""
    ids = {n["id"]: f"n{i}" for i, n in enumerate(g.get("nodes", []))}
    lines = [
        'digraph G { rankdir=LR; bgcolor="transparent"; overlap=false;',
        'node [fontname="Helvetica" fontsize=10 style="rounded,filled" fontcolor="white"];',
        'edge [fontname="Helvetica" fontsize=8 color="#a3a29c" fontcolor="#52514e" arrowsize=0.6];',
    ]
    for n in g.get("nodes", []):
        color = KIND_COLORS.get(n["kind"], IDLE)
        shape = {"document": "box", "chunk": "note", "entity": "ellipse"}.get(n["kind"], "box")
        extra = f"\\n{_dot_label(n['type'], 18)}" if n.get("type") else ""
        pen = ' penwidth=3 color="#0b0b0b"' if n.get("center") else f' color="{color}"'
        lines.append(f'{ids[n["id"]]} [label="{_dot_label(n["label"])}{extra}" shape={shape} '
                     f'fillcolor="{color}"{pen}];')  # fmt: skip
    for e in g.get("edges", []):
        if e["source"] in ids and e["target"] in ids:
            label = e["type"] + (f" ×{e['count']}" if e.get("count") else "")
            style = ' style="dashed"' if e.get("support") == "inferred" else ""
            lines.append(f'{ids[e["source"]]} -> {ids[e["target"]]} '
                         f'[label="{_dot_label(label, 24)}"{style}];')  # fmt: skip
    return "\n".join([*lines, "}"])


def eval_questions(eval_dir: Path) -> dict[str, dict[str, Any]]:
    path = eval_dir / "omnirag_eval_v1.json"
    return {q["id"]: q for q in load_json(path)["questions"]} if path.exists() else {}
