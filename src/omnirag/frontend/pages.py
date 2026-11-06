"""OmniRAG dashboard pages (Streamlit): see and try every retrieval feature, step by step.

Entry point: ``frontend/app.py`` (``uv run streamlit run src/omnirag/frontend/app.py``).

Pages:
* **Overview**: health of the API, Neo4j and Ollama; what is in the graph; the system's stages.
* **Retrieval lab**: one question through several retrieval modes side by side (dense,
  BM25, graph, fusions, reranking): where each chunk ranks in each mode, overlap, latency.
* **Workflow inspector**: the LangGraph workflow with every Phase 9 option (router,
  self-query, transforms, agent, reranking, context check, retries, per-document cap). Shows
  the path the run took, a timeline, what each node did, every ranked list and how RRF
  combined them into the final context.
* **Benchmarks**: the measured results in ``docs/benchmarks`` (read, never recomputed).

The dashboard only talks to the HTTP API (``frontend/api_client.py``); it holds no
database connection and no secrets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from omnirag.core.config import get_settings
from omnirag.frontend import views
from omnirag.frontend.api_client import ApiClient, ApiError

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR, BENCH_DIR = ROOT / "data" / "eval", ROOT / "docs" / "benchmarks"
MODES = ["dense", "bm25", "graph", "dense+bm25", "dense+bm25+graph", "dense+bm25+rerank",
         "dense+bm25+graph+rerank"]  # fmt: skip
ELEMENT_TYPES = ["paragraph", "list", "table", "record", "figure", "code"]
BLUE = views.RAN  # reference palette slot 1: one series per chart, so one hue
MODE_HELP = {
    "dense": "bge-m3 vectors, cosine similarity in the Neo4j vector index (meaning).",
    "bm25": "Lucene full-text, classic analyzer; identifiers like AUR-C103 are required.",
    "graph": "Links question words to entities, walks RELATED_TO edges (≤ 2 hops).",
    "dense+bm25": "Reciprocal Rank Fusion of dense and BM25: the measured default.",
    "dense+bm25+graph": "RRF of all three. Measured: lower MRR than dense+bm25.",
    "dense+bm25+rerank": "Fusion, then the bge-reranker-v2-m3 cross-encoder (10-17 s on CPU).",
    "dense+bm25+graph+rerank": "All three fused, then reranked.",
}


@st.cache_resource
def api() -> ApiClient:
    s = get_settings()
    return ApiClient(s.frontend_api_url, prefix=s.api_prefix, timeout_s=s.frontend_timeout_s)


@st.cache_data(ttl=600)
def examples() -> list[tuple[str, str]]:
    return views.example_questions(EVAL_DIR)


# ---------------------------------------------------------------- shared widgets
def question_input(key: str, default: str) -> str:
    labels = ["(type your own)", *[label for label, _ in examples()]]
    by_label = dict(examples())
    picked = st.selectbox("Example question from the eval sets", labels, key=f"{key}_pick",
                          index=next((i for i, lab in enumerate(labels)
                                      if by_label.get(lab) == default), 0))  # fmt: skip
    value = by_label.get(picked, st.session_state.get(f"{key}_last", default))
    question = st.text_input("Question", value=value, key=f"{key}_q_{picked}", max_chars=2000)
    st.session_state[f"{key}_last"] = question
    return question.strip()


def filters_input(key: str) -> dict[str, Any]:
    with st.expander("Metadata filters"):
        c1, c2, c3 = st.columns([2, 1, 1])
        types = c1.multiselect("Element types (empty = all)", ELEMENT_TYPES, key=f"{key}_types")
        latest = c2.toggle("Latest versions only", True, key=f"{key}_latest")
        no_gen = c3.toggle("Exclude model-written text", False, key=f"{key}_nogen",
                           help="Drop chunks containing image descriptions written by the "
                                "vision model (origin=generated).")  # fmt: skip
    return {"element_types": types or None, "latest_only": latest, "exclude_generated": no_gen}


def show_hit(h: dict[str, Any], *, expanded: bool = False) -> None:
    where = " > ".join(h.get("section_path") or []) or "—"
    pages = f" · pages {h['page_numbers']}" if h.get("page_numbers") else ""
    with st.expander(f"#{h['rank']}  {h['source_file']}  ·  {h['element_type']}  ·  "
                     f"score {h['score']:.4f}", expanded=expanded):  # fmt: skip
        st.caption(f"{where}{pages} · chunk `{h['chunk_id']}` · origins "
                   f"{', '.join(h.get('origins') or [])}")  # fmt: skip
        st.markdown(h["content"])
        if h.get("graph_paths"):
            st.caption("Graph paths: " + " | ".join(h["graph_paths"][:5]))
        if h.get("sources"):
            st.caption("Per-list ranks: " + ", ".join(
                f"{k} #{int(v['rank'])}" for k, v in h["sources"].items()))  # fmt: skip


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, fmt: str = ",.0f") -> alt.Chart:
    """Single-series horizontal bars: one hue, no legend (the title names the series),
    4px rounded data end, a tooltip per bar."""
    return (
        alt.Chart(df, title=title)
        .mark_bar(color=BLUE, cornerRadiusEnd=4, height=14)
        .encode(x=alt.X(f"{x}:Q", title=None, axis=alt.Axis(format=fmt, grid=True)),
                y=alt.Y(f"{y}:N", sort="-x", title=None,
                        axis=alt.Axis(labelOverlap=False, labelLimit=260)),
                tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)])
        .properties(height=alt.Step(26))
    )  # fmt: skip


# ---------------------------------------------------------------- pages
SYSTEM_DOT = """digraph S { rankdir=LR; bgcolor="transparent"; newrank=true;
node [shape=box style="rounded,filled" fillcolor="#2a78d6" color="#2a78d6" fontcolor="white"
      fontname="Helvetica" fontsize=11]; edge [color="#a3a29c" arrowsize=0.6];
files [label="Files\\nPDF · DOCX · XLSX · PPTX\\nHTML · images · CSV"];
ingest [label="Ingestion\\nloaders · OCR · tables\\nvision descriptions"];
chunk [label="Chunking\\nbge-m3 tokens\\n+ context"];
embed [label="Embeddings\\nbge-m3 (Ollama)"];
neo4j [label="Neo4j\\nchunks · vectors · full-text\\nentities · relations"];
retr [label="Retrieval\\ndense · BM25 · graph\\nRRF · rerank"];
wf [label="Workflow (LangGraph)\\nroute · transform · agent\\ncheck · retry"];
gen [label="Generation\\n(Phase 10)" style="rounded,dashed" fontcolor="#7a7974" color="#a3a29c"];
files -> ingest -> chunk -> embed -> neo4j;
question [label="Question" shape=box style="rounded" fontcolor="#7a7974" color="#a3a29c"];
question -> retr; neo4j -> retr [constraint=false]; retr -> wf -> gen;
{rank=same; files; question;} }"""  # two rows: indexing, question time. "graph" is a DOT keyword


def overview() -> None:
    st.title("OmniRAG")
    st.caption("Local-first multimodal GraphRAG. Everything runs on this machine: FastAPI, "
               "Neo4j, Ollama (qwen3:4b-instruct, bge-m3).")  # fmt: skip
    st.graphviz_chart(SYSTEM_DOT, width="stretch")
    try:
        health = api().health()
    except ApiError as exc:
        st.error(str(exc))
        st.code("uv run uvicorn omnirag.api.main:app --no-access-log", language="bash")
        return
    st.subheader("Health")
    comps = health.get("components", [])
    cols = st.columns(len(comps) + 1)
    cols[0].metric(
        "Overall",
        health.get("status", "?").upper(),
        border=True,
        help=f"{health.get('app')} {health.get('version')} · {health.get('environment')}",
    )
    for col, c in zip(cols[1:], comps, strict=False):
        lat = f" · {c['latency_ms']:.0f} ms" if c.get("latency_ms") is not None else ""
        col.metric(f"{c['name']}{lat}", c["status"].upper(), border=True,
                   help=c.get("error") or "Probe round-trip time in the label.")  # fmt: skip
    with st.expander("Component details (models loaded, VRAM, versions)"):
        st.json(comps, expanded=2)

    st.subheader("What is in the knowledge graph")
    try:
        stats = api().graph_stats()
    except ApiError as exc:
        st.warning(str(exc))
        return
    nodes, rels = stats.get("nodes", {}), stats.get("relationships", {})
    m = st.columns(5)
    m[0].metric("Documents", nodes.get("Document", 0), border=True)
    m[1].metric("Chunks", nodes.get("Chunk", 0), border=True, help="All chunk sets and versions.")
    m[2].metric(
        "With vectors",
        stats.get("chunks_with_vectors", 0),
        border=True,
        help=f"{stats.get('vector_dim')}-dim bge-m3 embeddings",
    )
    m[3].metric("Entities", nodes.get("Entity", 0), border=True)
    m[4].metric("Entity relations", rels.get("RELATED_TO", 0), border=True)
    c1, c2 = st.columns(2)
    c1.dataframe(
        pd.DataFrame(sorted(nodes.items()), columns=["node label", "count"]), hide_index=True
    )
    c2.dataframe(
        pd.DataFrame(sorted(rels.items()), columns=["relationship", "count"]), hide_index=True
    )
    try:
        docs = api().documents()["items"]
    except ApiError:
        docs = []
    if docs:
        st.subheader("Documents")
        st.dataframe(pd.DataFrame([{"file": d["source_file"], "kind": d["file_kind"],
                                    "status": d["status"], "elements": d["element_count"],
                                    "warnings": d["warning_count"], "id": d["document_id"]}
                                   for d in docs]), hide_index=True)  # fmt: skip


def retrieval_lab() -> None:
    st.title("Retrieval laboratory")
    st.caption("The same question through several retrieval modes. Each mode is one call to "
               "`POST /retrieval/search`; nothing is cached.")  # fmt: skip
    q = question_input("lab", "Why did cell AUR-C103 fail?")
    c1, c2 = st.columns([4, 1])
    modes = c1.multiselect("Modes", MODES, default=["dense", "bm25", "graph", "dense+bm25"],
                           key="lab_modes")  # fmt: skip
    k = c2.slider("k", 1, 20, 5, key="lab_k")
    with st.expander("What the modes do"):
        st.markdown("\n".join(f"- **{m}**: {MODE_HELP[m]}" for m in MODES))
    filters = filters_input("lab")
    if st.button("Run", type="primary", disabled=not (q and modes), key="lab_run"):
        results, errors = {}, {}
        with st.spinner(f"Running {len(modes)} modes…"):
            for mode in modes:
                try:
                    results[mode] = api().search(q, mode, k, filters)
                except ApiError as exc:
                    errors[mode] = str(exc)
        st.session_state["lab"] = {"q": q, "results": results, "errors": errors}
    run = st.session_state.get("lab")
    if not run:
        return
    for mode, err in run["errors"].items():
        st.error(f"{mode}: {err}")
    results = run["results"]
    if not results:
        return
    st.subheader("Where each chunk ranks in each mode")
    st.caption("Blank = that mode's top k doesn't contain the chunk. Rows sorted by best rank.")
    rows = views.compare_rows(results)
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_order=["file", *results, "type", "text", "chunk"],
                 column_config={m: st.column_config.NumberColumn(m, format="%d")
                                for m in results})  # fmt: skip
    c1, c2 = st.columns(2)
    lat = pd.DataFrame([{"mode": m, "ms": r["latency_ms"]} for m, r in results.items()])
    c1.altair_chart(bar_chart(lat, "ms", "mode", "Latency per mode (ms)"), width="stretch")
    c2.markdown("**Shared chunks between modes' top k**")
    c2.dataframe(pd.DataFrame(views.overlap(results)), hide_index=True)
    for tab, (mode, r) in zip(st.tabs(list(results)), results.items(), strict=False):
        with tab:
            st.caption(MODE_HELP.get(mode, ""))
            for h in r["hits"]:
                show_hit(h)
            with st.expander("What actually ran (details)"):
                st.json(r.get("details", {}), expanded=1)


PRESETS: dict[str, dict[str, Any]] = {
    # the measured defaults (ADR-0017): the agent only runs for multi-hop questions
    "Default": {"route": "rules", "self_query": "off", "agent": "auto", "transform": "none",
                "grade": True, "max_retries": 1, "rerank": False, "cap": 0},
    "No LLM calls": {"route": "off", "self_query": "off", "agent": "off", "transform": "none",
                     "grade": False, "max_retries": 0, "rerank": False, "cap": 0},
    "Everything on": {"route": "rules", "self_query": "soft", "agent": "always",
                      "transform": "multi_query", "grade": True, "max_retries": 1,
                      "rerank": True, "cap": 2},
}  # fmt: skip


def _apply(preset: str) -> None:
    for name, value in PRESETS[preset].items():
        st.session_state[f"wf_{name}"] = value


def strategy_input() -> tuple[dict[str, Any], int]:
    for name, value in PRESETS["Default"].items():
        st.session_state.setdefault(f"wf_{name}", value)
    st.session_state.setdefault("wf_steps", 2)
    p = st.columns(len(PRESETS) + 1)
    p[0].markdown("**Presets**")
    for col, name in zip(p[1:], PRESETS, strict=False):
        col.button(name, on_click=_apply, args=(name,), width="stretch", key=f"preset_{name}")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**1 · Before retrieval**")
        st.segmented_control(
            "Router",
            ["off", "rules", "llm"],
            key="wf_route",
            required=True,
            help="Rules: regexes, free. LLM: one call (~1 s); measured "
            "less accurate. Table / image routes add a preference list.",
        )
        st.segmented_control("Self-query", ["off", "soft", "hard"], key="wf_self_query",
                             required=True,
                             help="Documents the question names: a preference list (soft) "
                                  "or a filter (hard). Hard lost evidence on 2 relational "
                                  "questions.")  # fmt: skip
        st.selectbox("Query transform", ["none", "rewrite", "multi_query", "hyde",
                                         "multi_query+hyde"], key="wf_transform",
                     help="Measured in 9a: none helped both eval sets.")  # fmt: skip
    with b:
        st.markdown("**2 · Searching**")
        st.segmented_control("Agent", ["off", "auto", "always"], key="wf_agent", required=True,
                             help="auto = only when the router says multi_hop. The agent "
                                  "reads the passages and searches again.")  # fmt: skip
        st.slider("Agent steps (max)", 0, 3, key="wf_steps")
        st.toggle("Rerank (cross-encoder, CPU: +10-17 s)", key="wf_rerank")
    with c:
        st.markdown("**3 · Checking the context**")
        st.toggle("LLM grader after the identifier check", key="wf_grade")
        st.slider("Corrective retries (max)", 0, 3, key="wf_max_retries")
        st.slider("Max chunks per document (0 = no cap)", 0, 5, key="wf_cap")
        k = st.slider("k (passages returned)", 1, 15, 5, key="wf_k")
    s = st.session_state
    strategy = {"route": s.wf_route, "self_query": s.wf_self_query, "agent": s.wf_agent,
                "agent_steps": s.wf_steps, "transform": s.wf_transform, "grade": s.wf_grade,
                "max_retries": s.wf_max_retries, "rerank": s.wf_rerank,
                "max_per_document": s.wf_cap or None}  # fmt: skip
    return strategy, k


def _new_lists(details: dict[str, Any]) -> list[list[str]]:
    """The list keys each retrieve step added (lists keep insertion order)."""
    keys, out, i = list(details.get("lists", {})), [], 0
    for t in details.get("trace", []):
        if t["node"] == "retrieve":
            n = int(t.get("new_lists", 0))
            out.append(keys[i : i + n])
            i += n
    return out


def node_detail(t: dict[str, Any], d: dict[str, Any], counters: dict[str, int],
                added: list[list[str]], question: str) -> None:  # fmt: skip
    n = t["node"]
    i = counters.get(n, 0)
    counters[n] = i + 1
    st.caption(views.NODES.get(n, ""))
    if n == "route":
        r = d.get("route") or {}
        st.markdown(f"**Intent:** `{r.get('intent')}` via {r.get('method')}"
                    + (f" · cue: {', '.join(r['cues'])}" if r.get("cues") else ""))  # fmt: skip
        if r.get("element_types"):
            st.markdown(f"Adds a preference list restricted to {r['element_types']}.")
        sq = d.get("self_query")
        if sq:
            st.markdown(f"**Self-query:** model proposed {sq['proposed'] or 'nothing'}; "
                        f"accepted {sq['documents'] or 'none'}.")  # fmt: skip
            for rej in sq.get("rejected", []):
                st.markdown(f"- rejected `{rej.get('file')}`: {rej.get('reason')} "
                            f"(question matched {rej.get('matched')}, title words "
                            f"{rej.get('needs')})")  # fmt: skip
        st.markdown(f"**Agent:** {'on' if d['agent']['used'] else 'off'} for this question.")
    elif n == "transform":
        queries = [question, *t.get("queries", [])]
        st.markdown(
            f"**Transform `{t.get('transform')}` → queries:** "
            + ", ".join(f"`{x}`" for x in queries)
        )
        if d.get("hyde"):
            st.warning("HyDE passage (invented text, used only as a dense search key, never "
                       "as evidence):")  # fmt: skip
            st.write(d["hyde"])
        for e in t.get("errors", []):
            st.error(e)
    elif n == "retrieve":
        keys = added[i] if i < len(added) else []
        st.markdown(
            f"**{len(keys)} new ranked lists** this step; "
            f"{t.get('candidates')} candidates after fusing all {t.get('lists')}."
        )
        sub = {k: d["lists"][k] for k in keys if k in d.get("lists", {})}
        if sub:
            st.dataframe(pd.DataFrame(views.list_rows(sub, question)), hide_index=True)
    elif n == "plan":
        steps = d["agent"]["steps"]
        s = steps[i] if i < len(steps) else {}
        query = f" `{s['query']}`" if s.get("query") else ""
        st.markdown(f"**Action:** `{s.get('action')}`{query}")
        st.markdown(f"**Reason:** {s.get('reason') or '—'}")
        if s.get("refused"):
            st.error(f"Refused by the guard: {s['refused']}")
    elif n == "check":
        checks = d.get("checks", [])
        c = checks[i] if i < len(checks) else {}
        st.markdown(f"**Verdict:** `{c.get('verdict')}` via {c.get('method')}")
        if c.get("missing_identifiers"):
            st.markdown(f"Identifiers in no passage: {c['missing_identifiers']}")
        if c.get("missing"):
            st.markdown(f"**Missing:** {c['missing']}")
        if c.get("supporting"):
            st.markdown(f"Supporting chunks: {', '.join(c['supporting'])}")
    elif n == "correct":
        st.markdown(f"**New query:** `{t.get('query')}`")
    elif n == "rerank":
        st.markdown(f"{t.get('pairs')} pairs scored by the cross-encoder.")


def workflow_inspector() -> None:
    st.title("Workflow inspector")
    st.caption("One run of the LangGraph workflow (`POST /retrieval/advanced`), node by node. "
               "Pick a preset or set each option.")  # fmt: skip
    q = question_input("wf", "Who supplies the salt that the E-5 electrolyte is made with?")
    strategy, k = strategy_input()
    filters = filters_input("wf")
    if st.button("Run workflow", type="primary", disabled=not q, key="wf_run"):
        with st.spinner("Running the workflow (agent and grader calls take ~1-2 s each)…"):
            try:
                st.session_state["wf"] = {"q": q, "result": api().advanced(q, k, filters, strategy)}
            except ApiError as exc:
                st.session_state["wf"] = {"q": q, "error": str(exc)}
    run = st.session_state.get("wf")
    if not run:
        st.info("Tip: run the r02 question (who supplies the salt…) with the **Default** "
                "preset: the router sees a multi-hop question, the agent reads 'NaPF₆' in a "
                "passage and searches for its supplier.")  # fmt: skip
        return
    if "error" in run:
        st.error(run["error"])
        return
    r, d = run["result"], run["result"]["details"]
    trace = d.get("trace", [])

    m = st.columns(5)
    m[0].metric(
        "Verdict",
        d.get("verdict", "?"),
        border=True,
        help="From the last context check: what generation (Phase 10) will be told.",
    )
    m[1].metric("Total", f"{r['latency_ms'] / 1000:.2f} s", border=True)
    m[2].metric(
        "Nodes run",
        len(trace),
        border=True,
        help=f"LangGraph recursion limit for this strategy: {d.get('recursion_limit')}",
    )
    m[3].metric("Agent steps", len(d["agent"]["steps"]), border=True)
    m[4].metric("Ranked lists fused", len(d.get("lists", {})), border=True)

    st.subheader("Path through the graph")
    st.caption("Filled nodes ran (× times · total ms); solid edges were taken.")
    st.graphviz_chart(views.pipeline_dot(trace), width="stretch")

    st.subheader("Timeline")
    rows = views.timeline_rows(trace)
    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df, title="Time per node (ms)")
        .mark_bar(color=BLUE, cornerRadius=4, height=14)
        .encode(x=alt.X("start_ms:Q", title=None), x2="end_ms:Q",
                y=alt.Y("label:N", sort=list(df["label"]), title=None,
                        axis=alt.Axis(labelOverlap=False, labelLimit=200)),
                tooltip=["label", alt.Tooltip("ms:Q", format=",.0f"), "what happened"])
        .properties(height=alt.Step(26))
    )  # fmt: skip
    st.altair_chart(chart, width="stretch")

    st.subheader("Step by step")
    added, counters = _new_lists(d), {}
    for row, t in zip(rows, trace, strict=False):
        with st.expander(f"{row['step']}. **{t['node']}** · {row['ms']:,.0f} ms · "
                         f"{row['what happened']}"):  # fmt: skip
            node_detail(t, d, counters, added, run["q"])

    st.subheader("How the final ranking was built")
    st.caption("Reciprocal Rank Fusion: each list adds 1/(60 + rank) for every chunk it found. "
               "A chunk found by many lists, or near the top of one, wins.")  # fmt: skip
    st.dataframe(pd.DataFrame(views.contribution_rows(r["hits"],
                                                      d["strategy"].get("rrf_k", 60))),
                 hide_index=True)  # fmt: skip
    st.subheader(f"Final context ({len(r['hits'])} passages)")
    for h in r["hits"]:
        show_hit(h)
    with st.expander("Raw response (JSON)"):
        st.json(r, expanded=1)


def benchmarks() -> None:
    st.title("Benchmarks")
    st.caption("Measured results from `docs/benchmarks/*.json` (dates, models and versions are "
               "in each file). Nothing here is recomputed.")  # fmt: skip
    files = views.benchmark_files(BENCH_DIR)
    if not files:
        st.info(f"No benchmark files in {BENCH_DIR}")
        return
    names = [f.name for f in files]
    default = max((i for i, n in enumerate(names) if n.startswith("workflow")), default=0)
    name = st.selectbox("Result file", names, index=default)
    data = views.load_json(BENCH_DIR / name)
    st.caption(" · ".join(f"{k}: {data[k]}" for k in ("created", "dataset", "model", "k")
                          if k in data))  # fmt: skip
    rows = views.benchmark_rows(data)
    if not rows:
        st.json(data, expanded=1)
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    metric = st.segmented_control(
        "Chart metric",
        [m for m in views.METRICS if m in df],
        default="mrr" if "mrr" in df else None,
        required=True,
    )
    if metric:
        st.altair_chart(
            bar_chart(
                df[["config", metric]].dropna(),
                metric,
                "config",
                f"{metric} per configuration",
                ".3f",
            ),
            width="stretch",
        )
    per_q = {
        r["config"]: data["results"][r["config"]].get("per_question_recall")
        for r in rows
        if data["results"][r["config"]].get("per_question_recall")
    }
    if per_q:
        with st.expander("Per-question recall (wins and losses, not only means)"):
            st.dataframe(pd.DataFrame(per_q), width="stretch")


def main() -> None:
    st.set_page_config(page_title="OmniRAG", page_icon="🔎", layout="wide")
    page = st.navigation([
        st.Page(overview, title="Overview", icon="🏠", default=True),
        st.Page(retrieval_lab, title="Retrieval lab", icon="🧪", url_path="lab"),
        st.Page(workflow_inspector, title="Workflow inspector", icon="🧭", url_path="workflow"),
        st.Page(benchmarks, title="Benchmarks", icon="📊", url_path="benchmarks"),
    ])  # fmt: skip
    page.run()
