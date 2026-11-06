"""LangSmith tracing for the LangGraph workflow (Phase 12, off by default).

LangGraph traces every node to LangSmith when the ``langsmith`` SDK sees
``LANGSMITH_TRACING=true`` and an API key. Traces contain prompts, retrieved passages and
answers, i.e. document text, so OmniRAG only turns tracing on when
``OMNIRAG_LANGSMITH_TRACING=true`` *and* ``LANGSMITH_API_KEY`` is set, and by default asks the
SDK to hide inputs and outputs (``LANGSMITH_HIDE_INPUTS`` / ``LANGSMITH_HIDE_OUTPUTS``, read
by ``langsmith.Client``): the run tree, timings and errors are kept, the text is not.
``OMNIRAG_LANGSMITH_REDACT=false`` sends the full text (only for non-sensitive corpora).

Local observability does not depend on this: every chat turn's full trace is stored in
SQLite (``generation/history.py``) and shown by ``/analytics`` and the dashboard.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

log = structlog.get_logger(module=__name__)


def configure_tracing(settings: Any, environ: dict[str, str] | None = None) -> bool:
    """Set the LangSmith environment for this process. Returns whether tracing is on."""
    env = os.environ if environ is None else environ
    if not settings.langsmith_tracing:
        return False
    if not env.get("LANGSMITH_API_KEY"):
        log.warning("langsmith_tracing_disabled", reason="LANGSMITH_API_KEY is not set")
        return False
    env["LANGSMITH_TRACING"] = "true"
    env.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_redact:
        env["LANGSMITH_HIDE_INPUTS"] = "true"
        env["LANGSMITH_HIDE_OUTPUTS"] = "true"
    log.info("langsmith_tracing_enabled", project=env["LANGSMITH_PROJECT"],
             redacted=settings.langsmith_redact)  # fmt: skip
    return True
