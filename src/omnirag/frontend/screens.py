"""Dashboard screens added in Phases 10-14: chat with citations, the document library (upload,
ingestion status, preview), the knowledge-graph explorer, the evaluation dashboard and
performance analytics. Shared widgets live in ``pages.py``; pure helpers in ``views.py``.
Everything comes from the HTTP API; numbers are shown as the API reports them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from omnirag.frontend import views
from omnirag.frontend.api_client import ApiError
from omnirag.frontend.pages import EVAL_DIR, api, bar_chart

CHAT_PRESETS: dict[str, dict[str, Any] | None] = {
    "Default: router + agent + context check": None,
    "Fast: no LLM call before the answer": {"route": "off", "agent": "off", "grade": False,
                                            "max_retries": 0},
    "Quality: + cross-encoder reranking (slow on CPU)": {"rerank": True},
}  # fmt: skip
STARTERS = ["Why did cell AUR-C103 fail?",
            "Who supplies the salt that the E-5 electrolyte is made with?",
            "Which electrolyte swelled less at 45 °C, E-5 + FEC or E-7?",
            "What does the whiteboard photo say about the pilot batch?",
            "What is the warranty period of the Aurora S1?"]  # fmt: skip
TEXT_KINDS = {"txt", "md", "markdown", "csv", "json", "html"}
PAGES: dict[str, Any] = {}  # filled by pages.main(): lets a citation open the library


def _strategy() -> dict[str, Any] | None:
    preset = CHAT_PRESETS.get(st.session_state.get("chat_preset", ""))
    return None if preset is None else dict(preset)


# ---------------------------------------------------------------- chat
def render_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        a = turn["answer"]
        secs = (turn.get("timings") or {}).get("total_ms", 0) / 1000
        st.caption(f"{views.answer_badge(a)} · {secs:.1f} s")
        if a.get("answerable"):
            st.markdown(a["text"])
        if a.get("missing"):
            st.info(f"**Not in the sources:** {a['missing']}")
        if a.get("error"):
            st.error(a["error"])
        cards = views.citation_cards(turn)
        if cards:
            st.markdown("**Sources**")
        for c in cards:
            where = f" › {c['where']}" if c["where"] else ""
            pages = f" · p. {c['pages']}" if c["pages"] else ""
            with st.expander(f"[{c['number']}] {c['source_file']}{where}{pages}"):
                st.markdown(c["text"] or "_(passage text not stored)_")
                if c["claims"]:
                    st.caption("Supports: " + " · ".join(c["claims"]))
                st.caption(f"chunks `{', '.join(c['chunk_ids'])}` · origins "
                           f"{', '.join(c.get('origins') or [])}")  # fmt: skip
                if PAGES.get("library") and st.button(
                    "Open the document", key=f"open_{turn['id']}_{c['number']}"
                ):
                    st.session_state["lib_doc"] = c["document_id"]
                    st.switch_page(PAGES["library"])
        with st.expander("How this answer was checked"):
            rows = views.claim_rows(a)
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            r = turn.get("retrieval") or {}
            ctx = r.get("context") or {}
            st.caption(f"Context check: **{r.get('verdict')}** · route "
                       f"{(r.get('route') or {}).get('intent')} · context "
                       f"{ctx.get('tokens')}/{ctx.get('budget_tokens')} tokens in "
                       f"{len(ctx.get('blocks', []))} blocks · prompt "
                       f"{a.get('prompt_tokens')} → answer "
                       f"{a.get('eval_tokens')} tokens")  # fmt: skip
            t = turn.get("timings") or {}
            st.dataframe(pd.DataFrame([{"component": k, "ms": v} for k, v in t.items()]),
                         hide_index=True)  # fmt: skip


def _ask(question: str) -> None:
    s = st.session_state
    kwargs = {"conversation_id": s.get("chat_conv"), "strategy": _strategy(),
              "respect_check": s.get("chat_respect", True)}  # fmt: skip
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not s.get("chat_stream", True):
            with st.spinner("Retrieving and answering…"):
                turn = api().chat(question, **kwargs)
            s.chat_conv = turn["conversation_id"]
            return
        status = st.status("Retrieving…", expanded=True)
        claims_box, lines = st.empty(), []
        for ev in api().chat_stream(question, **kwargs):
            kind = ev.get("type")
            if kind == "start":
                s.chat_conv = ev["conversation_id"]
            elif kind == "retrieval":
                status.update(label=f"Retrieved {len(ev['blocks'])} sources in "
                                    f"{ev['latency_ms'] / 1000:.1f} s · check: {ev['verdict']} · "
                                    "writing the answer…")  # fmt: skip
                status.write(" · ".join(f"[{b['number']}] {b['source_file']}"
                                        for b in ev["blocks"]))  # fmt: skip
            elif kind == "claim":
                c = ev["claim"]
                marks = "".join(f"[{n}]" for n in c["sources"])
                lines.append(f"- {c['text']} {marks} — {views.status_label(c['status'])}")
                claims_box.markdown("\n".join(lines))
            elif kind == "error":
                status.update(label="Failed", state="error")
                st.error(ev.get("message"))
                return
            elif kind == "done":
                status.update(label=f"Done in {ev['timings']['total_ms'] / 1000:.1f} s",
                              state="complete", expanded=False)  # fmt: skip


def chat_page() -> None:
    st.title("Ask the documents")
    st.caption("Answers come only from the retrieved passages. Every claim carries a quote that "
               "code checks against the source; unsupported claims are removed, and questions "
               "the documents can't answer are refused.")  # fmt: skip
    s = st.session_state
    left, right = st.columns([1, 3], gap="large")
    with left:
        if st.button("➕ New conversation", key="chat_new", width="stretch"):
            s.chat_conv = None
        try:
            convs = api().history(30)
        except ApiError as exc:
            st.error(str(exc))
            return
        st.markdown("**Conversations**")
        for c in convs:
            label = f"{'▶ ' if c['id'] == s.get('chat_conv') else ''}{c['title'][:42]}"
            if st.button(f"{label} ({c['turns']})", key=f"conv_{c['id']}", width="stretch"):
                s.chat_conv = c["id"]
        with st.expander("Options", expanded=False):
            st.radio("Retrieval", list(CHAT_PRESETS), key="chat_preset")
            st.toggle("Stream claims as they are written", True, key="chat_stream")
            st.toggle("Refuse when no passage names the question's identifiers", True,
                      key="chat_respect")  # fmt: skip
            if s.get("chat_conv") and st.button("🗑 Delete this conversation", key="chat_del"):
                api().delete_conversation(s.chat_conv)
                s.chat_conv = None
                st.rerun()
    with right:
        conv = None
        if s.get("chat_conv"):
            try:
                conv = api().conversation(s.chat_conv)
            except ApiError:
                s.chat_conv = None
        for turn in (conv or {}).get("turns", []):
            render_turn(turn)
        starter = None
        if not conv:
            starter = st.pills("Try one", STARTERS, key="chat_starter")
        question = st.chat_input("Ask about the Aurora project documents…", key="chat_q")
        question = question or starter
        if question:
            try:
                _ask(question)
            except ApiError as exc:
                st.error(str(exc))
                return
            s.pop("chat_starter", None)
            st.rerun()


# ---------------------------------------------------------------- documents
def _preview(doc: dict[str, Any]) -> None:
    kind = doc.get("file_kind")
    try:
        data = api().original(doc["document_id"])
    except ApiError as exc:
        st.warning(f"Original not available: {exc}")
        return
    st.download_button("⬇ Download the original", data, file_name=doc["source_file"],
                       key=f"dl_{doc['document_id']}")  # fmt: skip
    if kind == "image":
        st.image(data, caption=doc["source_file"], width=480)
    elif kind in TEXT_KINDS:
        st.code(data[:20000].decode("utf-8", errors="replace"), language=None, wrap_lines=True)
    else:
        st.caption("Binary format: its extracted elements are below; download for the original.")


def library_page() -> None:
    st.title("Documents")
    s = st.session_state
    tab_lib, tab_up, tab_jobs = st.tabs(["Library", "Upload", "Ingestion status"])
    with tab_lib:
        try:
            docs = api().documents(200)["items"]
        except ApiError as exc:
            st.error(str(exc))
            return
        st.dataframe(pd.DataFrame([{"file": d["source_file"], "kind": d["file_kind"],
                                    "status": d["status"], "elements": d["element_count"],
                                    "warnings": d["warning_count"],
                                    "size KB": round(d["size_bytes"] / 1024, 1),
                                    "id": d["document_id"]} for d in docs]),
                     hide_index=True)  # fmt: skip
        ids = [d["document_id"] for d in docs]
        if not ids:
            st.info("No documents yet: upload some in the Upload tab.")
            return
        names = {d["document_id"]: d["source_file"] for d in docs}
        current = s.get("lib_doc") if s.get("lib_doc") in ids else ids[0]
        doc_id = st.selectbox("Preview a document", ids, index=ids.index(current),
                              format_func=lambda i: names[i], key="lib_pick")  # fmt: skip
        s.lib_doc = doc_id
        doc = api().document(doc_id)
        m = st.columns(4)
        m[0].metric("Kind", doc["file_kind"], border=True)
        m[1].metric("Status", doc["status"], border=True)
        m[2].metric("Elements", doc["element_count"], border=True)
        m[3].metric("Ingestion", f"{(doc.get('duration_ms') or 0) / 1000:.1f} s", border=True)
        if doc.get("warnings"):
            st.warning(" · ".join(doc["warnings"][:5]))
        _preview(doc)
        els = api().elements(doc_id, 500)["items"]
        types = sorted({e["element_type"] for e in els})
        pick = st.multiselect("Element types", types, key="lib_types")
        st.dataframe(pd.DataFrame([{"#": e["element_id"].rsplit("-", 1)[-1],
                                    "type": e["element_type"], "origin": e["origin"],
                                    "page": e.get("page_number"),
                                    "confidence": e.get("confidence"),
                                    "text": (e.get("content") or "")[:300]}
                                   for e in els if not pick or e["element_type"] in pick]),
                     hide_index=True)  # fmt: skip
        c1, c2 = st.columns(2)
        if c1.button("↻ Re-ingest this document", key="lib_reingest"):
            job = api().reingest(doc_id)
            st.success(f"Queued: {(job.get('job') or {}).get('job_id', 'already running')}")
        sure = c2.checkbox("I want to delete it (graph and files)", key="lib_sure")
        if c2.button("🗑 Delete", key="lib_delete", disabled=not sure):
            api().delete_document(doc_id)
            s.pop("lib_doc", None)
            st.rerun()
    with tab_up:
        files = st.file_uploader("Files to ingest (PDF, DOCX, XLSX, PPTX, HTML, MD, TXT, CSV, "
                                 "JSON, images)", accept_multiple_files=True,
                                 key="up_files")  # fmt: skip
        force = st.toggle("Re-process files that are already ingested", False, key="up_force")
        if files and st.button("Upload and ingest", type="primary", key="up_go"):
            for f in files:
                try:
                    r = api().upload(f.name, f.getvalue(), force=force)
                except ApiError as exc:
                    st.error(f"{f.name}: {exc}")
                    continue
                if r.get("duplicate"):
                    st.info(f"{f.name}: already ingested as `{r['document_id']}`")
                else:
                    st.success(f"{f.name}: job `{r['job']['job_id']}` queued "
                               f"(document `{r['document_id']}`)")  # fmt: skip
            st.caption("Follow the jobs in the Ingestion status tab.")
    with tab_jobs:
        if st.button("↻ Refresh", key="jobs_refresh"):
            st.rerun()
        try:
            stats, jobs = api().ingestion_status(), api().jobs(50)["items"]
        except ApiError as exc:
            st.error(str(exc))
            return
        counts = stats.get("counts", {})
        cols = st.columns(4)
        cols[0].metric(
            "Workers alive", f"{stats.get('workers_alive')}/{stats.get('workers')}", border=True
        )
        for col, key in zip(cols[1:], ("queued", "running", "completed"), strict=False):
            col.metric(key.capitalize(), counts.get(key, 0), border=True)
        st.dataframe(pd.DataFrame([{"job": j["job_id"], "file": j.get("source_file"),
                                    "status": j["status"], "document": j.get("document_id"),
                                    "s": round((j.get("duration_ms") or 0) / 1000, 1),
                                    "error": (j.get("error") or {}).get("message")}
                                   for j in jobs]),
                     hide_index=True)  # fmt: skip


# ---------------------------------------------------------------- graph explorer
def graph_page() -> None:
    st.title("Knowledge graph explorer")
    st.caption(
        "Documents, chunks and the entities extracted from them (validated against the "
        "text, Phase 5b). Solid edges are stated in the text, dashed ones inferred."
    )
    mode = st.segmented_control("View", ["Overview", "Entity", "Document"], default="Overview",
                                key="g_mode", required=True)  # fmt: skip
    limit = st.slider("Max rows", 10, 200, 60, key="g_limit")
    try:
        if mode == "Entity":
            ents = api().graph_entities(200)
            names = [e["name"] for e in ents]
            if not names:
                st.info("No entities yet: run `python -m omnirag.knowledge_base extract`.")
                return
            name = st.selectbox(
                "Entity",
                names,
                key="g_entity",
                format_func=lambda n: next(
                    f"{e['name']} · {e['type']} · {e['mentions']} mentions"
                    for e in ents
                    if e["name"] == n
                ),
            )
            g = api().graph_neighborhood(entity=name, limit=limit)
        elif mode == "Document":
            docs = api().documents(200)["items"]
            names = {d["document_id"]: d["source_file"] for d in docs}
            doc_id = st.selectbox("Document", list(names), format_func=lambda i: names[i],
                                  key="g_doc")  # fmt: skip
            g = api().graph_neighborhood(document_id=doc_id, limit=limit)
        else:
            g = api().graph_neighborhood(limit=limit)
    except ApiError as exc:
        st.error(str(exc))
        return
    legend = " · ".join(f"<span style='color:{c}'>■</span> {k}"
                        for k, c in views.KIND_COLORS.items())  # fmt: skip
    st.markdown(f"{len(g['nodes'])} nodes · {len(g['edges'])} edges &nbsp;&nbsp; {legend}",
                unsafe_allow_html=True)  # fmt: skip
    if g["nodes"]:
        st.graphviz_chart(views.graph_dot(g), width="stretch")
    if mode == "Entity" and g["nodes"]:
        detail = api().graph_entity(st.session_state["g_entity"])
        st.subheader(detail["entity"].get("name", ""))
        rel = detail.get("relations", [])
        if rel:
            st.dataframe(pd.DataFrame([{"relation": r["type"], "direction": r["direction"],
                                        "other": r["other"], "support": r["support"],
                                        "evidence": r.get("evidence")} for r in rel]),
                         hide_index=True)  # fmt: skip
        st.caption(
            f"{len(detail.get('mentions', []))} mentions · same as: "
            + (", ".join(x["name"] for x in detail.get("same_as", [])) or "—")
        )
    with st.expander("Nodes and edges as tables"):
        st.dataframe(pd.DataFrame(g["nodes"]), hide_index=True)
        st.dataframe(pd.DataFrame(g["edges"]), hide_index=True)


# ---------------------------------------------------------------- evaluation
@st.cache_data(ttl=600)
def _questions() -> dict[str, dict[str, Any]]:
    return views.eval_questions(EVAL_DIR)


def evaluation_page() -> None:
    st.title("Evaluation")
    st.caption("Measured results only: every number comes from a run of "
               "`python -m omnirag.evaluation.rag_eval` (or `POST /evaluate`) over the "
               "110-question set. Nothing here is recomputed or estimated.")  # fmt: skip
    try:
        runs = api().eval_runs()
    except ApiError as exc:
        st.error(str(exc))
        return
    with st.expander("Start a new evaluation run"):
        exps = st.multiselect("Experiments", [f"E{i}" for i in range(1, 8)], ["E3", "E7"],
                              key="ev_exps")  # fmt: skip
        limit = st.number_input("First N questions (0 = all 110)", 0, 110, 10, key="ev_limit")
        judge = st.toggle("LLM judge", True, key="ev_judge")
        if st.button("Start", key="ev_start", disabled=not exps):
            try:
                r = api().evaluate(exps, int(limit) or None, judge)
                st.success(f"Started `{r['run_id']}`: it runs in the background.")
            except ApiError as exc:
                st.error(str(exc))
    done = [r for r in runs if r["status"] == "completed"]
    active = [r for r in runs if r["status"] == "running"]
    for r in active:
        state = api().eval_run(r["run_id"])
        st.info(f"Running `{r['run_id']}`: " + (state.get("progress") or ["starting…"])[-1])
    if not done:
        st.info("No completed evaluation yet.")
        return
    labels = {
        r["run_id"]: f"{r['run_id']} · {r.get('created', '')[:16]} · "
        f"{', '.join(r.get('experiments') or [])}"
        for r in done
    }
    run_id = st.selectbox("Report", list(labels), format_func=lambda i: labels[i], key="ev_run")
    report = api().eval_run(run_id, full=True)["report"]
    m = st.columns(4)
    m[0].metric("Questions", report.get("questions"), border=True,
                help=f"{report.get('dataset')} (10 unanswerable)")  # fmt: skip
    m[1].metric("Generator", report.get("generation_model"), border=True)
    m[2].metric("Prompt", report.get("prompt_version"), border=True)
    if report.get("wall_s"):
        m[3].metric("Run time", f"{report['wall_s'] / 60:.0f} min", border=True)
    else:
        m[3].metric("Run", "stopped early" if report.get("stopped_early") else "—", border=True)
    if report.get("not_measured"):
        st.warning(
            f"Not measured in this report: {', '.join(report['not_measured'])}. "
            "Experiments marked *partial* cover only the first questions of the set: "
            "compare them on the same questions (docs/benchmarks/"
            "same76-rag-eval-2026-11-06.json), not with the full-set rows."
        )
    if report.get("dataset_errata"):
        with st.expander("Dataset errata applied when scoring"):
            for e in report["dataset_errata"]:
                st.markdown(f"**{e['date']}** · {', '.join(e['questions'])}: {e['change']}")
    rows = views.eval_rows(report)
    st.subheader("Experiments compared")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    col = st.segmented_control(
        "Chart a metric",
        [c for c in views.EVAL_COLUMNS],
        default="answer correct",
        key="ev_metric",
        required=True,
    )
    data = pd.DataFrame(views.metric_rows(report, col))
    if not data.empty:
        fmt = ",.0f" if col.endswith("ms") else ".3f"
        st.altair_chart(bar_chart(data, "value", "experiment", col, fmt), width="stretch")
    st.subheader("Where questions fail (root cause)")
    st.caption("The first pipeline stage that lost a fact (evaluation/failures.py). Later "
               "stages aren't blamed again.")  # fmt: skip
    fr = pd.DataFrame(views.failure_rows(report))
    if fr.empty:
        st.success("No failures recorded.")
    else:
        st.dataframe(fr.pivot_table(index="category", columns="experiment", values="questions",
                                    fill_value=0), width="stretch")  # fmt: skip
    st.subheader("Per question")
    exp = st.selectbox("Experiment", [r["exp"] for r in rows], key="ev_exp",
                       index=len(rows) - 1)  # fmt: skip
    only = st.toggle("Only failed questions", True, key="ev_only")
    pq = views.per_question_rows(report, exp, _questions())
    if only:
        pq = [r for r in pq if r["correct"] == "❌" or r["hallucination"]]
    st.dataframe(pd.DataFrame(pq), hide_index=True)
    judge_info = report["experiments"][exp].get("judge", {})
    st.caption(f"LLM judge ({report.get('judge_model')}): {judge_info.get('correctness')} · "
               f"agreement with the deterministic answer check "
               f"{judge_info.get('agreement_with_answer_check')}")  # fmt: skip


# ---------------------------------------------------------------- performance
def performance_page() -> None:
    st.title("Performance and answer quality")
    try:
        a = api().analytics()
    except ApiError as exc:
        st.error(str(exc))
        return
    c = a.get("chat") or {}
    st.subheader("Live chat traffic")
    st.caption("From the stored trace of every chat turn (SQLite). No ground truth here: "
               "quality signals are what the checks themselves report.")  # fmt: skip
    m = st.columns(5)
    m[0].metric("Turns", c.get("turns", 0), border=True)
    m[1].metric("Answered", c.get("answered", 0), border=True)
    m[2].metric("Grounded", c.get("grounded", 0), border=True,
                help="Answered with no unsupported claim.")  # fmt: skip
    m[3].metric("Refused", sum((c.get("refused_by") or {}).values()), border=True,
                help=str(c.get("refused_by")))  # fmt: skip
    m[4].metric("Citation accuracy", c.get("citation_accuracy_mean") or "—", border=True)
    lat = pd.DataFrame(views.latency_rows(a))
    if not lat.empty:
        st.altair_chart(bar_chart(lat, "p50 ms", "component", "Latency p50 per component (ms)"),
                        width="stretch")  # fmt: skip
        st.dataframe(lat, hide_index=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Failure flags**")
    c1.dataframe(pd.DataFrame(sorted((c.get("failure_flags") or {}).items()),
                              columns=["flag", "turns"]), hide_index=True)  # fmt: skip
    c2.markdown("**Context check verdicts**")
    c2.dataframe(pd.DataFrame(sorted((c.get("verdicts") or {}).items()),
                              columns=["verdict", "turns"]), hide_index=True)  # fmt: skip
    if c.get("recent"):
        st.markdown("**Recent turns**")
        st.dataframe(pd.DataFrame(c["recent"]), hide_index=True)
    ev = a.get("evaluation")
    if not ev:
        return
    st.subheader(f"Measured in the latest evaluation ({ev.get('questions')} questions)")
    rows = [{"exp": f"{k} · {e['label']}",
             **{f"{part} {p}": e["performance"].get(f"{key}_{p}")
                for part, key in (("end-to-end", "end_to_end_ms"), ("retrieval", "retrieval_ms"),
                                  ("embedding", "embed_ms"), ("rerank", "rerank_ms"),
                                  ("generation", "generation_ms")) for p in ("p50", "p95")},
             "q/min": e["performance"].get("throughput_q_per_min")}
            for k, e in ev["experiments"].items()]  # fmt: skip
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    try:
        latest = next(r for r in api().eval_runs() if r["status"] == "completed")
        report = api().eval_run(latest["run_id"])["report"]
    except (ApiError, StopIteration):
        return
    if report.get("memory"):
        st.markdown("**Memory during the run**")
        st.json(report["memory"], expanded=True)
    if report.get("indexing"):
        idx = report["indexing"]
        st.markdown(f"**Indexing:** {idx['documents']} documents, {idx['chunks']} chunks · "
                    f"ingestion {idx['ingestion_ms_total'] / 1000:.1f} s · graph load "
                    f"{idx['graph_load_ms_total'] / 1000:.1f} s")  # fmt: skip
