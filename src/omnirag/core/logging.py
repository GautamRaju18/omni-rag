"""Structured logging.

We use structlog so every log line is a JSON object (machine-parseable, ready for later
observability work) and so request-scoped values such as ``request_id`` are attached
automatically via ``contextvars`` — they follow the request across ``await`` points.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structlog (our logs) and stdlib logging (library logs) once at startup."""
    log_level = logging.getLevelNamesMapping()[level.upper()]

    # Third-party libraries (neo4j driver, httpx, uvicorn) log via stdlib logging.
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too noisy

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,  # pulls in request_id etc.
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
