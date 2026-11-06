"""Answer, claim and citation models (Phases 10-11).

An answer is a list of **claims**. Each claim carries the model's own words, the source
numbers it cited and a quote copied from a source; code then checks the quote against the
context (``generation/citations.py``) and records what it found. Nothing here is trusted
because the model said so.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Support = Literal["direct", "inferred"]
# verified:    the quote is in a cited source, and every number in the claim is too
# relocated:   the quote is in a source the model didn't cite (citation corrected, flagged)
# partial:     no verbatim quote, but the claim's numbers/identifiers are in the cited sources
# unsupported: nothing checks out (a hallucination candidate); never shown as an answer
ClaimStatus = Literal["verified", "relocated", "partial", "unsupported"]


class Citation(BaseModel):
    """A source block the answer cites, with everything needed to show and verify it."""

    number: int  # the [n] in the answer text = the block's position in the context
    source_file: str
    document_id: str
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    chunk_ids: list[str]
    element_ids: list[str] = Field(default_factory=list)
    origins: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    text: str
    quote: str = ""
    support: Support = "direct"
    model_sources: list[int] = Field(default_factory=list)  # what the model cited
    sources: list[int] = Field(default_factory=list)  # after validation (what we show)
    status: ClaimStatus = "unsupported"
    quote_found_in: list[int] = Field(default_factory=list)  # blocks containing the quote
    generated_evidence: bool = False  # rests on model-written text (image description)
    issues: list[str] = Field(default_factory=list)


class AnswerMetrics(BaseModel):
    claims: int = 0
    verified: int = 0
    relocated: int = 0
    partial: int = 0
    unsupported: int = 0
    citation_accuracy: float | None = None  # claims whose cited source holds the quote
    faithfulness: float | None = None  # (verified + relocated) / claims


class Answer(BaseModel):
    question: str
    answerable: bool
    text: str  # the answer shown to the user, with [n] markers; unsupported claims removed
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)  # only the cited blocks
    missing: str | None = None  # what the sources don't contain
    refused_by: Literal["check", "model", "empty_context", "error"] | None = None
    grounded: bool = False  # answerable, ≥1 claim, and no unsupported claim
    metrics: AnswerMetrics = Field(default_factory=AnswerMetrics)
    model: str = ""
    prompt_version: str = ""
    prompt_tokens: int = 0
    eval_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None  # the model's reply as parsed (for debugging)
    error: str | None = None
