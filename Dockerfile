# syntax=docker/dockerfile:1
# OmniRAG API + dashboard image (Phase 15, ADR-0021). One image, two commands:
#   API:        uvicorn omnirag.api.main:app (default CMD)
#   dashboard:  streamlit run src/omnirag/frontend/app.py
# Ollama stays on the host (it owns the GPU); Neo4j is its own container. No secrets are baked
# in: they come from .env at runtime (docker compose env_file).

FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

FROM python:3.11.15-slim-bookworm AS build
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached layer), from the committed lock: exact versions, CPU-only torch.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.11.15-slim-bookworm
# Tesseract for OCR (scans, images); GL/GLib for Docling's image pipeline; curl for healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgl1 \
       libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 1000 --create-home omnirag
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app/src /app/src
COPY data/eval /app/data/eval
COPY docs/benchmarks /app/docs/benchmarks
COPY .streamlit /app/.streamlit
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 OMNIRAG_DATA_DIR=/data \
    OMNIRAG_BENCHMARKS_DIR=/app/docs/benchmarks HF_HOME=/data/hf
USER omnirag
EXPOSE 8000 8501
CMD ["uvicorn", "omnirag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
