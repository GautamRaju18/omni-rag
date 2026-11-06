"""Retrieval CLI.

uv run python -m omnirag.retrieval search "Why did cell AUR-C103 fail?"
    [--mode dense|bm25|graph|dense+bm25|dense+bm25+graph|dense+bm25+rerank|...+graph+rerank]
    [--k 5] [--doc DOC_ID ...] [--file NAME ...] [--type table ...] [--no-generated]
    [--all-versions] [--identifiers terms|boost|require] [--full]
"""

from __future__ import annotations

import argparse
import json
import sys

from omnirag.core.config import get_settings
from omnirag.core.logging import configure_logging
from omnirag.retrieval import build_retrieval
from omnirag.retrieval.models import RetrievalFilters


def cmd_search(args: argparse.Namespace) -> int:
    service = build_retrieval(get_settings())
    filters = RetrievalFilters(
        latest_only=not args.all_versions, document_ids=args.doc or None,
        source_files=args.file or None, element_types=args.type or None,
        exclude_generated=args.no_generated,
    )  # fmt: skip
    try:
        if args.mode == "bm25" and args.identifiers:
            bm25 = service.retrievers["bm25"]
            result = bm25.retrieve(args.query, args.k, filters, identifier_mode=args.identifiers)
        else:
            result = service.search(args.query, args.mode, args.k, filters)
    finally:
        service.close()
    print(json.dumps(result.details, ensure_ascii=False))
    for hit in result.hits:
        section = " > ".join(hit.section_path)
        print(f"\n#{hit.rank}  {hit.score:.4f}  {hit.source_file}  {section}  "
              f"[{hit.element_type}; {'/'.join(hit.origins)}]  {hit.chunk_id}")  # fmt: skip
        text = hit.content if args.full else hit.content[:240].replace("\n", " ")
        print(f"    {text}")
        if hit.sources:
            print("    via " + ", ".join(f"{n} #{s['rank']:.0f}" for n, s in hit.sources.items()))
        for path in hit.graph_paths[:4]:
            print(f"    graph: {path}")
    print(f"\n{len(result.hits)} hits in {result.latency_ms} ms")
    return 0


def cmd_advanced(args: argparse.Namespace) -> int:
    from omnirag.workflows.retrieval_graph import Strategy, build_workflow

    settings = get_settings()
    service = build_retrieval(settings)
    workflow = build_workflow(settings, service)
    strategy = workflow.default.model_copy(update={
        "transform": args.transform, "grade": not args.no_grade,
        "max_retries": args.retries, "rerank": args.rerank, "route": args.route,
        "self_query": args.self_query, "agent": args.agent, "agent_steps": args.agent_steps,
        "max_per_document": args.max_per_document, "context_budget": args.budget,
        "expand": args.expand, "compress": args.compress})  # fmt: skip
    try:
        result = workflow.retrieve(args.query, args.k, strategy=Strategy(**strategy.model_dump()))
    finally:
        workflow.close()
        service.close()
    d = result.details
    print(f"verdict={d['verdict']} attempts={d['attempts']} queries={d['queries']}")
    route = d["route"]
    print(
        f"  route: {route['intent']} via {route['method']} {route['cues']}"
        + (f" -> prefers {route['element_types']}" if route["element_types"] else "")
    )
    if d["self_query"]:
        sq = d["self_query"]
        print(
            f"  self-query: documents={sq['documents']} proposed={sq['proposed']}"
            + (f" rejected={[r.get('file') for r in sq['rejected']]}" if sq["rejected"] else "")
        )
    for step in d["agent"]["steps"]:
        print(
            f"  agent: {step['action']} {step['query']!r}"
            + (f" REFUSED ({step['refused']})" if step["refused"] else f" ({step['reason']})")
        )
    for check in d["checks"]:
        print(f"  check #{check['attempt']}: {check['verdict']} via {check['method']}"
              + (f" (missing: {check['missing']})" if check["missing"] else ""))  # fmt: skip
    print("  trace: " + " -> ".join(f"{t['node']} {t['ms']:.0f}ms" for t in d["trace"]))
    for hit in result.hits:
        text = hit.content if args.full else hit.content[:200].replace("\n", " ")
        print(f"\n#{hit.rank}  {hit.source_file}  {' > '.join(hit.section_path)}\n    {text}")
    print(f"\n{len(result.hits)} hits in {result.latency_ms} ms")
    ctx = d.get("context")
    if ctx:
        print(
            f"\ncontext for generation: {len(ctx['blocks'])} blocks, {ctx['tokens']}/"
            f"{ctx['budget_tokens']} tokens (expand {ctx['options']['expand']}, compress "
            f"{ctx['options']['compress']}), {len(ctx['skipped'])} chunks didn't fit"
        )
        if args.show_context:
            print("\n" + ctx["rendered"])
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging("WARNING", json_logs=False)
    parser = argparse.ArgumentParser(prog="python -m omnirag.retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument(
        "--mode",
        default=get_settings().retrieval_default_mode,
        choices=[
            "dense",
            "bm25",
            "graph",
            "dense+bm25",
            "dense+bm25+graph",
            "dense+bm25+rerank",
            "dense+bm25+graph+rerank",
        ],
    )
    p.add_argument("--k", type=int, default=get_settings().retrieval_k)
    p.add_argument("--doc", action="append", help="restrict to a document ID (repeatable)")
    p.add_argument("--file", action="append", help="restrict to a source file name")
    p.add_argument("--type", action="append", help="element type: paragraph, table, figure, ...")
    p.add_argument("--no-generated", action="store_true", help="skip chunks with model text")
    p.add_argument("--all-versions", action="store_true")
    p.add_argument("--identifiers", choices=["terms", "boost", "require"], default=None)
    p.add_argument("--full", action="store_true", help="print whole chunks")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser(
        "advanced", help="LangGraph workflow: route, transforms, agent, check, retries"
    )
    p.add_argument("query")
    p.add_argument("--k", type=int, default=get_settings().retrieval_k)
    p.add_argument(
        "--transform",
        default=get_settings().workflow_transform,
        choices=["none", "rewrite", "multi_query", "hyde", "multi_query+hyde"],
    )
    p.add_argument("--no-grade", action="store_true")
    p.add_argument("--retries", type=int, default=get_settings().workflow_max_retries)
    p.add_argument("--rerank", action="store_true")
    p.add_argument(
        "--route", default=get_settings().workflow_route, choices=["off", "rules", "llm"]
    )
    p.add_argument(
        "--self-query", default=get_settings().workflow_self_query, choices=["off", "soft", "hard"]
    )
    p.add_argument(
        "--agent", default=get_settings().workflow_agent, choices=["off", "auto", "always"]
    )
    p.add_argument("--agent-steps", type=int, default=get_settings().workflow_agent_steps)
    p.add_argument("--max-per-document", type=int, default=get_settings().workflow_max_per_document)
    p.add_argument("--budget", type=int, default=get_settings().context_budget_tokens)
    p.add_argument("--expand", default=get_settings().context_expand,
                   choices=["none", "neighbors", "section"])  # fmt: skip
    p.add_argument("--compress", default=get_settings().context_compress,
                   choices=["none", "lexical", "embedding", "llm"])  # fmt: skip
    p.add_argument("--show-context", action="store_true", help="print the assembled context")
    p.add_argument("--full", action="store_true")
    p.set_defaults(fn=cmd_advanced)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
