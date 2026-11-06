"""Self-query (Phase 9b): turn a document the question names into a metadata filter.

"What does the safety review recommend for pilot-line pouches?" names a document; "Who owns
risk R2?" doesn't. The LLM proposes, code decides (the Phase 5b pattern):

1. The **catalog** (latest documents: file name and title) comes from the graph. The reply
   schema's enum *is* the catalog, so the model can only name files that exist.
2. A proposal is accepted only when the question contains the document's **descriptor**:
   at least two of its distinctive title/file-name words, or its only one ("whiteboard").
   Distinctive = not a stop word, not a number, not in most titles ("aurora", "project").
   Words match by prefix from 4 letters, so "scanned" matches "scan".
   This rejects the model picking a document because it might *contain* the answer.

Accepted documents become a preference (``soft``: one extra ranked list restricted to them
joins RRF; nothing is filtered out) or a filter (``hard``: only they are searched, so a wrong
pick loses the evidence). Which one is the default is a measured decision (ADR-0017).
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from omnirag.knowledge_base.entities import norm
from omnirag.llm.chat import LLMError, OllamaChat

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_STOP = frozenset(
    "a an and the of for in on at to by with from is are was this that it its as or project "
    "pdf csv json html docx xlsx pptx jpg png md txt".split()  # untitled files: title = name
)
# Generic words name a document only next to a distinctive one: alone, "notes" or "page"
# would tie half the corpus to any question that uses them. Kind words come from the format.
_GENERIC = frozenset("results notes note report review".split())
_KIND = {
    "html": "page website web", "xlsx": "spreadsheet workbook sheet",
    "csv": "spreadsheet table", "pptx": "slides slide presentation deck",
    "jpg": "photo image picture", "png": "photo image picture", "docx": "document",
}  # fmt: skip

SELF_QUERY_SYSTEM = (
    "You map a question to the documents it explicitly refers to. You get a catalog of "
    "documents (file name and title). Return the file names of the documents that the "
    "question itself names or clearly describes (by title, kind of document or what it "
    "shows). Return an empty list when the question asks about a topic without referring to "
    "a specific document. Never pick a document only because it might contain the answer."
)


def _words(text: str) -> list[str]:
    return _TOKEN.findall(norm(text.replace("_", " ")))


def descriptors(catalog: list[dict[str, Any]]) -> dict[str, tuple[set[str], set[str]]]:
    """(distinctive, generic) words per source file, from its title, file stem and format."""
    raw = {}
    for d in catalog:
        stem, _, ext = d["source_file"].rpartition(".")
        raw[d["source_file"]] = (
            {w for w in _words(f"{d.get('title') or ''} {stem}")
             if len(w) > 1 and any(c.isalpha() for c in w)},
            set(_KIND.get(ext.lower(), "").split()),
        )  # fmt: skip
    df = Counter(w for ws, _ in raw.values() for w in ws)
    common = {w for w, n in df.items() if n > len(catalog) / 2}
    out = {}
    for f, (ws, kind) in raw.items():
        distinctive = {w for w in ws if w not in common and w not in _STOP and w not in _GENERIC}
        out[f] = (distinctive, (ws & _GENERIC) | kind)
    return out


def _mentions(word: str, tokens: list[str]) -> bool:
    return any(t == word or (len(word) >= 4 and t.startswith(word)) for t in tokens)


def grounded(question: str, words: tuple[set[str], set[str]]) -> tuple[bool, list[str]]:
    """(the question names the document, the descriptor words it contains).

    Named = at least one distinctive word, and two descriptor words in all unless the
    document has a single distinctive word ("whiteboard").
    """
    distinctive, generic = words
    tokens = _words(question)
    hit_d = sorted(w for w in distinctive if _mentions(w, tokens))
    hit_g = sorted(w for w in generic if _mentions(w, tokens))
    return bool(hit_d) and len(hit_d) + len(hit_g) >= min(2, len(distinctive)), hit_d + hit_g


@dataclass
class SelfQueryResult:
    documents: list[dict[str, Any]] = field(default_factory=list)  # accepted: id + file
    proposed: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)  # file + reason
    method: str = "llm"
    ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"documents": [d["source_file"] for d in self.documents],
                "proposed": self.proposed, "rejected": self.rejected,
                "method": self.method, "ms": self.ms}  # fmt: skip


class SelfQuery:
    def __init__(self, llm: OllamaChat, catalog: Callable[[], list[dict[str, Any]]]) -> None:
        self.llm = llm
        self.catalog = catalog  # () -> [{document_id, source_file, title}], latest only

    def run(self, question: str) -> SelfQueryResult:
        started = time.perf_counter()
        catalog = self.catalog()
        if not catalog:
            return SelfQueryResult(method="no_catalog")
        files = [d["source_file"] for d in catalog]
        schema = {"type": "object", "required": ["documents"], "properties": {"documents": {
            "type": "array", "items": {"type": "string", "enum": files}}}}  # fmt: skip
        listing = "\n".join(f"- {d['source_file']}: {d.get('title') or ''}" for d in catalog)
        try:
            r = self.llm.chat(SELF_QUERY_SYSTEM, f"Catalog:\n{listing}\n\nQuestion: {question}",
                              schema=schema, num_predict=128)  # fmt: skip
        except LLMError as exc:  # no filter is always safe
            return SelfQueryResult(method="llm_error", rejected=[{"reason": str(exc)[:200]}])
        proposed = list(dict.fromkeys(f for f in r.data.get("documents", []) if f in files))
        words = descriptors(catalog)
        by_file = {d["source_file"]: d for d in catalog}
        out = SelfQueryResult(
            proposed=proposed, ms=round((time.perf_counter() - started) * 1000, 1)
        )
        for f in proposed:
            ok, hit = grounded(question, words[f])
            if ok:
                out.documents.append({"document_id": by_file[f]["document_id"], "source_file": f})
            else:
                out.rejected.append({"file": f, "reason": "not named in the question",
                                     "matched": hit,
                                     "needs": sorted(words[f][0])})  # fmt: skip
        return out


def catalog_from(store: Any) -> Callable[[], list[dict[str, Any]]]:
    def load() -> list[dict[str, Any]]:
        return store.query(
            "MATCH (d:Document {is_latest: true}) RETURN d.document_id AS document_id, "
            "d.source_file AS source_file, d.title AS title ORDER BY source_file"
        )

    return load
