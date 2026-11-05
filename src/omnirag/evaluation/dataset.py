"""Evaluation questions with verbatim evidence.

A question has ``facts``; each fact is a list of **alternative** evidence quotes (the same fact
often appears in several documents). Relevance is decided by text containment, the same way
for every chunking strategy and size:

* a chunk *covers* a fact when its content contains any alternative (whitespace-normalised);
* ``resolve()`` also maps every quote to the element(s) it comes from (table rows → the
  TABLE_ROW element), so ground truth is available as element IDs (ADR-0008) and every quote
  is verified to really exist in the named file.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from omnirag.ingestion.models import DocumentElement, ElementType


class Evidence(BaseModel):
    file: str
    text: str


class EvalQuestion(BaseModel):
    id: str
    type: str
    question: str
    answer: str | None
    facts: list[list[Evidence]] = Field(default_factory=list)
    evidence_origin: str = "original"  # "generated": only model-written text holds the fact

    @property
    def answerable(self) -> bool:
        return bool(self.facts)


class EvalDataset(BaseModel):
    name: str
    version: int
    created: str
    description: str
    questions: list[EvalQuestion]

    @classmethod
    def load(cls, path: Path) -> EvalDataset:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def norm(text: str) -> str:
    return " ".join(text.split())


def covered_facts(content: str, question: EvalQuestion) -> set[int]:
    """Indexes of the question's facts that ``content`` contains."""
    text = norm(content)
    return {i for i, alts in enumerate(question.facts) if any(norm(a.text) in text for a in alts)}


def resolve(
    question: EvalQuestion, elements_by_file: dict[str, list[DocumentElement]]
) -> list[set[str]]:
    """Element IDs per fact. Raises ``LookupError`` if a quote is not in its file."""
    out = []
    for alts in question.facts:
        ids: set[str] = set()
        for alt in alts:
            found = _find(norm(alt.text), elements_by_file.get(alt.file, []))
            if not found:
                raise LookupError(f"{question.id}: {alt.text!r} not found in {alt.file}")
            ids |= found
        out.append(ids)
    return out


def _find(quote: str, elements: list[DocumentElement]) -> set[str]:
    rows = {
        (e.parent_element_id, e.metadata.get("row_index")): e.element_id
        for e in elements
        if e.element_type is ElementType.TABLE_ROW
    }
    hits: set[str] = set()
    for e in elements:
        if e.element_type is ElementType.TABLE:
            for i, line in enumerate(e.content.split("\n")[2:]):  # Markdown data rows
                if quote in norm(line):
                    hits.add(rows.get((e.element_id, i), e.element_id))
            if not hits and quote in norm(e.content):
                hits.add(e.element_id)
        elif e.element_type is not ElementType.TABLE_ROW and quote in norm(e.content):
            hits.add(e.element_id)
    return hits
