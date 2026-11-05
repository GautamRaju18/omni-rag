"""Entity and relation extraction with the local chat model (Ollama structured output).

Per chunk, one ``/api/chat`` call with ``format`` set to a JSON schema. Ollama then constrains
decoding, so the reply always parses and every type is one of the enums. The schema says
nothing about *truth*, though: validation against the source text is done afterwards by
``entities.validate_chunk``.

* The model sees the chunk's contextual header (document title > section) to disambiguate,
  but entities must be grounded in the chunk text itself.
* ``temperature 0`` and a fixed seed make reruns repeatable for the same model digest.
* Raw replies are stored with the validated result. A chunk whose prompt input, model
  digest and prompt version are unchanged is not sent again, so validation rules can change
  and be re-applied (``revalidate``) without any model call.

Measured 2026-11-05 (Ollama 0.40.14, qwen3:4b-instruct, RTX 3060 6 GB): 12–22 s per chunk,
440–940 output tokens. That's too slow to run on every upload by default
(``extraction_enabled=false``); run it as a batch step.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from omnirag.chunking.models import Chunk
from omnirag.ingestion.models import DocumentElement, DocumentRecord
from omnirag.knowledge_base.entities import (
    ChunkExtraction,
    DocumentExtraction,
    EntityType,
    Rejection,
    RelationType,
    resolve_document,
    validate_chunk,
)

log = structlog.get_logger(module=__name__)

PROMPT_VERSION = "kg-v2"  # v2: endpoints must be listed; no status values
SYSTEM_PROMPT = """You extract a knowledge graph from one chunk of a document.

Entities: only things explicitly named in the TEXT: people, organizations, projects, products,
materials, test samples, milestones, documents. Copy each name exactly as written in the TEXT.
Do not extract numbers, dates, units, measurements, percentages, status values
(passed, in_progress) or generic words.
Entity types: PERSON, ORGANIZATION, PROJECT, PRODUCT, MATERIAL (chemicals, electrolytes,
components), SAMPLE (individual test cells or specimens), MILESTONE, DOCUMENT.

Relations: only between entities you extracted, and only when the TEXT states them.
Every name used in a relation must also be listed in entities.
The source is the subject of the statement:
- A PART_OF B: A belongs to B.
- P RESPONSIBLE_FOR X: person or organization P owns, leads or is in charge of X.
- P MEMBER_OF O: P works for or belongs to organization or project O.
- P AUTHORED D: P wrote or prepared document D.
- A USES M: sample, product or material A contains or is made with material M.
- S TESTED_BY P: sample S was tested by person P.
- A REPLACES B: A is the new one that takes the place of B.
- A DEPENDS_ON B: milestone, project or product A needs B.
- O SUPPLIES M: organization O supplies material M.
- A RELATED_TO B: a stated link that fits none of the types above.
evidence: copy the exact sentence or table row from the TEXT that states the relation.
Do not guess and do not use outside knowledge."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"},
                           "type": {"type": "string", "enum": [t.value for t in EntityType]}},
            "required": ["name", "type"]}},
        "relations": {"type": "array", "items": {
            "type": "object",
            "properties": {"source": {"type": "string"},
                           "type": {"type": "string", "enum": [t.value for t in RelationType]},
                           "target": {"type": "string"}, "evidence": {"type": "string"}},
            "required": ["source", "type", "target", "evidence"]}},
    },
    "required": ["entities", "relations"],
}  # fmt: skip


class ExtractionError(RuntimeError):
    """The model server is unreachable or the model is missing."""


def prompt_input(chunk: Chunk) -> str:
    header = f"{chunk.context}\n\n" if chunk.context else ""
    return f"{header}TEXT:\n{chunk.content}"


def input_key(text: str, model_digest: str) -> str:
    material = f"{PROMPT_VERSION}\x1f{model_digest}\x1f{text}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class OllamaExtractor:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 300.0,
        num_ctx: int = 8192,
        num_predict: int = 2048,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self._digest: str | None = None
        self.stats: dict[str, float] = {"calls": 0, "cached": 0, "seconds": 0.0, "tokens": 0}

    @property
    def model_digest(self) -> str:
        if self._digest is None:
            try:
                tags = self._client.get("/api/tags").raise_for_status().json()
            except httpx.HTTPError as exc:
                raise ExtractionError(f"Ollama unreachable: {exc}") from exc
            entry = next((m for m in tags.get("models", []) if m["name"] == self.model), None)
            if entry is None:
                raise ExtractionError(f"Model {self.model!r} not found (ollama pull it)")
            self._digest = entry["digest"]
        return self._digest

    def ask(self, text: str) -> dict[str, Any]:
        """One structured-output call; one retry if the reply isn't a JSON object."""
        payload = {
            "model": self.model, "stream": False, "format": SCHEMA, "keep_alive": "5m",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": text}],
            "options": {"temperature": 0, "seed": 42, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict},
        }  # fmt: skip
        last: Exception | None = None
        for _ in range(2):
            started = time.perf_counter()
            try:
                response = self._client.post("/api/chat", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ExtractionError(f"Ollama chat failed: {exc}") from exc
            body = response.json()
            self.stats["calls"] += 1
            self.stats["seconds"] += time.perf_counter() - started
            self.stats["tokens"] += body.get("eval_count", 0)
            try:
                parsed = json.loads(body["message"]["content"])
                if isinstance(parsed, dict):
                    return parsed
                last = ValueError(f"reply is a {type(parsed).__name__}, not an object")
            except (KeyError, json.JSONDecodeError) as exc:  # e.g. cut off at num_predict
                last = exc
        raise ValueError(f"unusable model reply: {last}")

    def extract_document(
        self,
        record: DocumentRecord,
        elements: list[DocumentElement],
        chunks: list[Chunk],
        config_id: str,
        previous: DocumentExtraction | None = None,
    ) -> DocumentExtraction:
        digest = self.model_digest
        by_id = {e.element_id: e for e in elements}
        reusable = {}
        if (
            previous
            and previous.model_digest == digest
            and previous.prompt_version == PROMPT_VERSION
        ):
            reusable = {c.chunk_id: c for c in previous.chunks if c.raw is not None}
        results = []
        for chunk in (c for c in chunks if c.retrievable):
            text = prompt_input(chunk)
            key = input_key(text, digest)
            old = reusable.get(chunk.chunk_id)
            if old is not None and old.input_sha256 == key:
                raw, error = old.raw, None
                self.stats["cached"] += 1
            else:
                try:
                    raw, error = self.ask(text), None
                except ValueError as exc:
                    raw, error = None, str(exc)
            result = ChunkExtraction(chunk_id=chunk.chunk_id, input_sha256=key, raw=raw,
                                     error=error)  # fmt: skip
            if error:
                result.rejected.append(Rejection(kind="output", item={}, reason="model_error"))
            else:
                own = [by_id[i] for i in chunk.element_ids if i in by_id]
                result.mentions, result.relations, result.rejected = validate_chunk(chunk, own, raw)
            results.append(result)
        resolve_document(results)
        extraction = DocumentExtraction(
            document_id=record.document_id, config_id=config_id, model=self.model,
            model_digest=digest, prompt_version=PROMPT_VERSION,
            extracted_at=datetime.now(UTC).isoformat(), chunks=results,
        )  # fmt: skip
        log.info("extraction_done", document_id=record.document_id, **extraction.counts())
        return extraction


class ExtractionStep:
    """Pipeline step: extract one chunk set of a document (reusing unchanged raw replies)."""

    def __init__(self, extractor: OllamaExtractor, config_id: str) -> None:
        self.extractor = extractor
        self.config_id = config_id

    def run(
        self,
        record: DocumentRecord,
        elements: list[DocumentElement],
        sets: dict[str, list[Chunk]],
        previous: DocumentExtraction | None = None,
    ) -> DocumentExtraction | None:
        chunks = sets.get(self.config_id)
        if not chunks:
            return None
        return self.extractor.extract_document(record, elements, chunks, self.config_id, previous)


def revalidate(
    extraction: DocumentExtraction, elements: list[DocumentElement], chunks: list[Chunk]
) -> DocumentExtraction:
    """Re-apply the current validation rules to stored raw replies (no model calls)."""
    by_chunk = {c.chunk_id: c for c in chunks}
    by_id = {e.element_id: e for e in elements}
    for result in extraction.chunks:
        chunk = by_chunk.get(result.chunk_id)
        if chunk is None or result.error:
            continue
        own = [by_id[i] for i in chunk.element_ids if i in by_id]
        result.mentions, result.relations, result.rejected = validate_chunk(chunk, own, result.raw)
    resolve_document(extraction.chunks)
    return extraction
