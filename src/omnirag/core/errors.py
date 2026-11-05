"""Domain exceptions and their mapping to HTTP responses.

Library code raises ``OmniRAGError`` subclasses (no FastAPI imports needed in the domain
layers); the API layer translates them into a consistent JSON error envelope.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

log = structlog.get_logger(module=__name__)


class OmniRAGError(Exception):
    """Base class for all expected application errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DependencyUnavailableError(OmniRAGError):
    """An external dependency (Neo4j, Ollama, ...) is unreachable or unhealthy."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "dependency_unavailable"


class ValidationFailedError(OmniRAGError):
    """Input passed schema validation but failed a domain rule."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_failed"


class FileValidationError(ValidationFailedError):
    """File is empty, too large, corrupt, or its content does not match its extension."""

    code = "invalid_file"


class UnsupportedFileTypeError(OmniRAGError):
    """The file extension is not one OmniRAG ingests."""

    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_file_type"


class LoaderNotAvailableError(OmniRAGError):
    """The file type is recognized but its loader is not installed or implemented yet."""

    status_code = status.HTTP_501_NOT_IMPLEMENTED
    code = "loader_not_available"


class NotFoundError(OmniRAGError):
    """A requested resource (document, job, ...) does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class PayloadTooLargeError(OmniRAGError):
    """Upload exceeds ``OMNIRAG_MAX_UPLOAD_MB``."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"


class ConflictError(OmniRAGError):
    """The request conflicts with current state (e.g. deleting a document being ingested)."""

    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class IngestionQueueFullError(OmniRAGError):
    """Too many queued ingestion jobs; the client should retry later."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "ingestion_queue_full"


def _envelope(request: Request, code: str, message: str, details: dict) -> dict:
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    return {
        "error": {"code": code, "message": message, "details": details, "request_id": request_id}
    }


async def _omnirag_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, OmniRAGError)  # noqa: S101 - registered only for this type
    log.warning("request_failed", code=exc.code, error=exc.message, path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(request, exc.code, exc.message, exc.details),
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error", path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_envelope(request, "internal_error", "Internal server error", {}),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the error handlers to an app instance."""
    app.add_exception_handler(OmniRAGError, _omnirag_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
