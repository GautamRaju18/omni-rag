"""Grounded answer generation over the assembled context (Phase 10).

``AnswerGenerator`` takes the workflow's result (``details.context`` from Phase 9c, plus the
context check's verdict) and returns a validated ``Answer``:

* **No context, no call.** An empty context, or a context check that found the question's
  identifier nowhere ("insufficient"), returns a refusal without asking the model: a 4B model
  given unrelated passages is the classic hallucination setup. ``respect_check=False`` turns
  this off (measured in Phase 12).
* **One structured call** (``prompts.ANSWER_SCHEMA``): claims with sources, quotes and
  direct/inferred support, then ``citations.validate`` checks every one against the blocks.
* **Streaming** (``stream``): the same call with ``stream=True``. The JSON arrives token by
  token; ``ClaimStream`` cuts out each claim object as soon as it is complete, so the client
  gets *validated* claims one at a time, then the final answer.

Model errors never raise: they become an answer with ``refused_by="error"`` (the API reports
it), because a chat turn must always have a record.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

import structlog

from omnirag.generation.citations import render_answer, validate, validate_claim
from omnirag.generation.models import Answer
from omnirag.generation.prompts import ANSWER_SCHEMA, ANSWER_SYSTEM, PROMPT_VERSION, user_prompt
from omnirag.llm.chat import LLMError
from omnirag.retrieval.context import ContextBlock

log = structlog.get_logger(module=__name__)

_CLAIMS_START = re.compile(r'"claims"\s*:\s*\[')


class ClaimStream:
    """Incrementally extract complete objects from the reply's ``claims`` array."""

    def __init__(self) -> None:
        self.buf = ""
        self.pos: int | None = None
        self.depth = 0
        self.in_string = False
        self.escaped = False
        self.start: int | None = None
        self.closed = False

    def feed(self, delta: str) -> list[dict[str, Any]]:
        self.buf += delta
        out: list[dict[str, Any]] = []
        if self.pos is None:
            m = _CLAIMS_START.search(self.buf)
            if not m:
                return out
            self.pos = m.end()
        while not self.closed and self.pos < len(self.buf):
            c = self.buf[self.pos]
            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif c == "\\":
                    self.escaped = True
                elif c == '"':
                    self.in_string = False
            elif c == '"':
                self.in_string = True
            elif c in "{[":
                if self.depth == 0 and c == "{":
                    self.start = self.pos
                self.depth += 1
            elif c in "}]":
                if self.depth == 0:  # the claims array itself closed
                    self.closed = True
                else:
                    self.depth -= 1
                    if self.depth == 0 and c == "}" and self.start is not None:
                        try:
                            out.append(json.loads(self.buf[self.start : self.pos + 1]))
                        except json.JSONDecodeError:
                            pass
                        self.start = None
            self.pos += 1
        return out


def blocks_of(context: dict[str, Any] | None) -> list[ContextBlock]:
    return [ContextBlock.model_validate(b) for b in (context or {}).get("blocks", [])]


class AnswerGenerator:
    def __init__(self, llm: Any, *, num_predict: int = 1024, respect_check: bool = True) -> None:
        self.llm = llm
        self.num_predict = num_predict
        self.respect_check = respect_check

    # ---------------------------------------------------------------- shared
    def _refusal(self, question: str, verdict: str | None, checks: list[dict] | None,
                 blocks: list[ContextBlock], respect_check: bool) -> Answer | None:  # fmt: skip
        base = {"question": question, "answerable": False, "text": "",
                "model": self.llm.model, "prompt_version": PROMPT_VERSION}  # fmt: skip
        if not blocks:
            return Answer(**base, refused_by="empty_context",
                          missing="No passages were retrieved for this question.")  # fmt: skip
        if respect_check and verdict == "insufficient":
            last = (checks or [{}])[-1]
            ids = last.get("missing_identifiers") or []
            missing = (
                f"No source mentions {', '.join(ids)}."
                if ids
                else last.get("missing") or "The retrieved passages don't answer it."
            )
            return Answer(**base, refused_by="check", missing=missing)  # fmt: skip
        return None

    def _finish(self, question: str, data: Any, blocks: list[ContextBlock], started: float,
                prompt_tokens: int, eval_tokens: int) -> Answer:  # fmt: skip
        data = data if isinstance(data, dict) else {}
        claims, citations, metrics = validate(data.get("claims") or [], blocks)
        said_answerable = bool(data.get("answerable")) and bool(claims)
        missing = str(data.get("missing") or "").strip() or None
        shown = [c for c in claims if c.status != "unsupported"]
        answerable = said_answerable and bool(shown)
        return Answer(
            question=question, answerable=answerable, text=render_answer(claims),
            claims=claims, citations=citations, missing=missing,
            refused_by=None if answerable else "model",
            grounded=answerable and metrics.unsupported == 0, metrics=metrics,
            model=self.llm.model, prompt_version=PROMPT_VERSION,
            prompt_tokens=prompt_tokens, eval_tokens=eval_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000, 1), raw=data,
        )  # fmt: skip

    def _error(self, question: str, exc: Exception, started: float) -> Answer:
        log.warning("generation_failed", error=str(exc)[:200])
        return Answer(question=question, answerable=False, text="", refused_by="error",
                      error=str(exc)[:500], model=self.llm.model, prompt_version=PROMPT_VERSION,
                      latency_ms=round((time.perf_counter() - started) * 1000, 1))  # fmt: skip

    # ---------------------------------------------------------------- run
    def generate(self, question: str, context: dict[str, Any] | None, verdict: str | None = None,
                 checks: list[dict] | None = None, *,
                 respect_check: bool | None = None) -> Answer:  # fmt: skip
        started = time.perf_counter()
        blocks = blocks_of(context)
        respect = self.respect_check if respect_check is None else respect_check
        if (refusal := self._refusal(question, verdict, checks, blocks, respect)) is not None:
            return refusal
        try:
            r = self.llm.chat(ANSWER_SYSTEM, user_prompt(question, context["rendered"]),
                              schema=ANSWER_SCHEMA, num_predict=self.num_predict)  # fmt: skip
        except LLMError as exc:
            return self._error(question, exc, started)
        return self._finish(question, r.data, blocks, started, r.prompt_tokens, r.eval_tokens)

    def stream(self, question: str, context: dict[str, Any] | None, verdict: str | None = None,
               checks: list[dict] | None = None, *,
               respect_check: bool | None = None) -> Iterator[dict[str, Any]]:  # fmt: skip
        """Events: ``{"type": "claim", "claim": {...}}`` per validated claim as it completes,
        then ``{"type": "answer", "answer": {...}}`` (the authoritative, fully parsed one)."""
        started = time.perf_counter()
        blocks = blocks_of(context)
        respect = self.respect_check if respect_check is None else respect_check
        if (refusal := self._refusal(question, verdict, checks, blocks, respect)) is not None:
            yield {"type": "answer", "answer": refusal.model_dump(mode="json")}
            return
        parser, text, final = ClaimStream(), "", {}
        try:
            for event in self.llm.stream(
                ANSWER_SYSTEM,
                user_prompt(question, context["rendered"]),
                schema=ANSWER_SCHEMA,
                num_predict=self.num_predict,
            ):
                if "delta" in event:
                    text += event["delta"]
                    for raw in parser.feed(event["delta"]):
                        claim = validate_claim(raw, blocks)
                        if claim.text:
                            yield {"type": "claim", "claim": claim.model_dump(mode="json")}
                else:
                    final = event
        except LLMError as exc:
            yield {"type": "answer", "answer": self._error(question, exc, started).model_dump(
                mode="json")}  # fmt: skip
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            answer = self._error(question, LLMError(f"unusable JSON: {exc}"), started)
        else:
            answer = self._finish(question, data, blocks, started, final.get("prompt_tokens", 0),
                                  final.get("eval_tokens", 0))  # fmt: skip
        yield {"type": "answer", "answer": answer.model_dump(mode="json")}
