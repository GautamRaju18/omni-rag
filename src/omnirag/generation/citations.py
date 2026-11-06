"""Citation and answer validation (Phase 11): the model proposes, code checks.

For each claim the model returns (text, cited source numbers, quote, direct/inferred):

1. **Source numbers** outside the context are dropped (recorded): a citation is never made up.
2. **Quote**: normalised (NFKC, dashes, case, whitespace, no markdown) and searched in every
   block. Found in a cited block → ``verified``; found only in another block → ``relocated``
   (the citation is corrected to where the words really are, and flagged). A quote that leaves
   out columns of a real table row ("Category: 300 | AUR-C106: 96.2") counts if its parts
   occur in order within one line (``match`` → "elided", flagged).
3. **Numbers**: every number in the claim's text must occur in the claim's sources. A verified
   quote with a different number in the paraphrase is the dangerous hallucination ("swelled
   by 15%" quoting "swelled by 14%"), so it drops to ``partial`` with an issue.
4. No quote match → ``partial`` if the claim's numbers and identifiers all occur in the cited
   blocks and most of its content words do; otherwise ``unsupported``.
5. **Direct vs inferred**: "direct" needs a verified/relocated quote; a quote inside a
   generated image description makes the claim "inferred" and ``generated_evidence``.

Unsupported claims are removed from the answer text but kept in the response (hallucination
detection, Phase 12 metrics).
"""

from __future__ import annotations

import re
from typing import Any

from omnirag.generation.models import AnswerMetrics, Citation, Claim
from omnirag.knowledge_base.entities import norm
from omnirag.retrieval.context import ContextBlock
from omnirag.retrieval.sparse import identifiers

_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_WORD = re.compile(r"[a-z][a-z\-]{2,}")
_STOP = frozenset(
    "the and for with that this from are was were has have had its their which what when where "
    "who whom whose why how into than then also been being after before about over under all "
    "any each per not but only both other such".split()
)
_GENERATED = "image description (generated"
_ELISION = re.compile(r"\s*(?:\||…|\.\.\.)\s*")
MIN_QUOTE = 8  # characters, after normalisation: shorter "quotes" ("E-7") prove nothing
PARTIAL_OVERLAP = 0.5  # share of the claim's content words found in its sources


def numbers(text: str) -> set[float]:
    """Numbers in the text ("240,000" = "240000"; digits inside identifiers like AUR-C103
    don't count: they belong to the identifier)."""
    out = set()
    for raw in _NUMBER.findall(_THOUSANDS.sub("", norm(text))):
        try:
            out.add(float(raw.replace(",", ".")))
        except ValueError:  # pragma: no cover - the regex only matches numbers
            continue
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(norm(text)) if w not in _STOP}


def citation_of(number: int, block: ContextBlock) -> Citation:
    return Citation(number=number, source_file=block.source_file, document_id=block.document_id,
                    section_path=block.section_path, page_numbers=block.page_numbers,
                    chunk_ids=block.chunk_ids, element_ids=block.element_ids,
                    origins=block.origins)  # fmt: skip


def _segments(quote: str) -> list[str]:
    """An elided quote's parts: "category: 300 | aur-c106: 96.2" → two segments."""
    return [s for s in (p.strip(" |") for p in _ELISION.split(quote)) if len(s) >= 2]


def match(quote: str, text: str) -> str | None:
    """``"verbatim"`` if the normalised quote is in the text; ``"elided"`` if the quote's
    parts (split at "|", "…", "...") all occur, in order, within one line of the text (the
    model left out columns or words of a real line); else ``None``."""
    if len(quote) < MIN_QUOTE:
        return None
    if quote in norm(text):
        return "verbatim"
    parts = _segments(quote)
    if len(parts) < 2 or sum(len(p) for p in parts) < MIN_QUOTE:
        return None
    for line in text.split("\n"):
        n, at = norm(line), 0
        for p in parts:
            at = n.find(p, at)
            if at < 0:
                break
            at += len(p)
        else:
            return "elided"
    return None


def _original_text(block_text: str) -> str:
    """The block without its generated image-description lines."""
    return "\n".join(line for line in block_text.split("\n")
                     if not norm(line).startswith(_GENERATED))  # fmt: skip


def validate_claim(raw: dict[str, Any], blocks: list[ContextBlock]) -> Claim:
    text = " ".join(str(raw.get("text", "")).split())
    quote = " ".join(str(raw.get("quote", "")).split())
    support = raw.get("support") if raw.get("support") in ("direct", "inferred") else "inferred"
    asked = [n for n in raw.get("sources", []) if isinstance(n, int)]
    claim = Claim(text=text, quote=quote, support=support, model_sources=asked)
    valid = [n for n in dict.fromkeys(asked) if 1 <= n <= len(blocks)]
    if len(valid) < len(set(asked)):
        claim.issues.append(f"cited sources not in the context: {sorted(set(asked) - set(valid))}")
    texts = {n: norm(blocks[n - 1].text) for n in range(1, len(blocks) + 1)}

    q = norm(quote)
    how = {n: match(q, blocks[n - 1].text) for n in texts}
    found = [n for n, m in how.items() if m]
    claim.quote_found_in = found
    if any(how[n] == "elided" for n in found):
        claim.issues.append("quote leaves out parts of the source line")
    if found and set(found) & set(valid):
        claim.status, claim.sources = "verified", valid
    elif found:
        claim.status, claim.sources = "relocated", [found[0]]
        claim.issues.append(f"quote is in source {found[0]}, not in {valid or 'none'}")
    else:
        claim.sources = valid
        if q and len(q) < MIN_QUOTE:
            claim.issues.append("quote too short to verify")
        elif q:
            claim.issues.append("quote not found verbatim in any source")
        else:
            claim.issues.append("no quote")

    cited = " ".join(texts[n] for n in claim.sources)
    missing_numbers = sorted(numbers(text) - numbers(cited))
    if claim.status in ("verified", "relocated"):
        if missing_numbers:
            claim.status = "partial"
            claim.issues.append(f"numbers not in the cited sources: {missing_numbers}")
    elif claim.sources:
        ids_ok = all(norm(i) in cited for i in identifiers(text))
        words = _content_words(text)
        overlap = len(words & _content_words(cited)) / len(words) if words else 0.0
        if not missing_numbers and ids_ok and overlap >= PARTIAL_OVERLAP:
            claim.status = "partial"
        else:
            claim.status = "unsupported"
            if missing_numbers:
                claim.issues.append(f"numbers not in the cited sources: {missing_numbers}")
            if not ids_ok:
                claim.issues.append("identifiers not in the cited sources")
    else:
        claim.status = "unsupported"

    if claim.status in ("verified", "relocated") and not any(  # only model-written text has it
        match(q, _original_text(blocks[n - 1].text)) for n in claim.quote_found_in
    ):
        claim.generated_evidence = True
        if claim.support == "direct":
            claim.support = "inferred"
            claim.issues.append("evidence is a generated image description")
    if claim.support == "direct" and claim.status not in ("verified", "relocated"):
        claim.support = "inferred"
    return claim


def validate(raw_claims: list[Any], blocks: list[ContextBlock]
             ) -> tuple[list[Claim], list[Citation], AnswerMetrics]:  # fmt: skip
    claims = [validate_claim(c, blocks) for c in raw_claims if isinstance(c, dict)]
    claims = [c for c in claims if c.text]
    used = sorted({n for c in claims if c.status != "unsupported" for n in c.sources})
    citations = [citation_of(n, blocks[n - 1]) for n in used]
    m = AnswerMetrics(claims=len(claims))
    for c in claims:
        setattr(m, c.status, getattr(m, c.status) + 1)
    if claims:
        held = sum(1 for c in claims if set(c.quote_found_in) & set(c.model_sources))
        m.citation_accuracy = round(held / len(claims), 3)
        m.faithfulness = round((m.verified + m.relocated) / len(claims), 3)
    return claims, citations, m


def render_answer(claims: list[Claim]) -> str:
    """The user-facing answer: supported claims with their [n] markers, in model order."""
    parts = []
    for c in claims:
        if c.status == "unsupported":
            continue
        sentence = c.text if c.text.endswith((".", "!", "?")) else c.text + "."
        marks = "".join(f"[{n}]" for n in c.sources)
        parts.append(f"{sentence[:-1]} {marks}{sentence[-1]}" if marks else sentence)
    return " ".join(parts)
