"""Span-based text assembly and splitting.

Everything works on **character spans into one assembled text**, never on copied substrings. A
chunk's span can then be mapped back to exactly the elements (``Segment``s) it overlaps, and we
know whether each element is contained fully or only partly. That's provenance that survives
any splitting strategy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from omnirag.chunking.tokenizer import Span, TokenCounter

# Recursive split order: paragraph > line > sentence > clause > word, then a hard token cut.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass
class Segment:
    """One source element's text inside an assembled text."""

    start: int
    end: int
    element_id: str
    element_type: str
    page_number: int | None
    section_path: list[str]
    origin: str
    confidence: float


@dataclass
class Assembled:
    text: str
    segments: list[Segment] = field(default_factory=list)

    def add(self, text: str, segment: Segment | None = None, sep: str = "\n\n") -> None:
        """Append ``text``; ``segment`` (start/end ignored) records which element it is."""
        if self.text and text:
            self.text += sep
        start = len(self.text)
        self.text += text
        if segment is not None:
            segment.start, segment.end = start, len(self.text)
            self.segments.append(segment)

    def heading_ends(self) -> frozenset[int]:
        return frozenset(s.end for s in self.segments if s.element_type in ("heading", "title"))

    def overlapping(self, span: Span) -> list[tuple[Segment, bool]]:
        """Segments overlapping ``span`` and whether each is only partly inside it."""
        a, b = span
        return [
            (seg, not (a <= seg.start and seg.end <= b))
            for seg in self.segments
            # Zero-length segments (an IMAGE with no text) count when they sit in the span.
            if (seg.start < b and seg.end > a) or (seg.start == seg.end and a <= seg.start <= b)
        ]


def strip_span(text: str, span: Span) -> Span:
    a, b = span
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def token_windows(text: str, counter: TokenCounter, size: int, overlap: int) -> list[Span]:
    """Fixed windows of exactly ``size`` tokens, stepping ``size - overlap`` (baseline)."""
    offsets = counter.offsets(text)
    step = size - overlap
    spans = []
    for i in range(0, len(offsets), step):
        group = offsets[i : i + size]
        # Re-tokenizing a cut-out slice can give one token more than the slice of tokens did
        # (a word cut at the start splits differently). Measured: 9 of 50 windows at 128
        # came out as 129. Trim until the slice itself fits.
        while len(group) > 1 and counter.count(text[group[0][0] : group[-1][1]]) > size:
            group = group[:-1]
        spans.append((group[0][0], group[-1][1]))
        if i + size >= len(offsets):
            break
    return spans


def split_spans(
    text: str,
    span: Span,
    counter: TokenCounter,
    max_tokens: int,
    overlap: int = 0,
    separators: Sequence[str] = SEPARATORS,
    keep_with_next: frozenset[int] = frozenset(),
) -> list[Span]:
    """Recursive splitting: windows of at most ``max_tokens`` that end at the coarsest boundary
    possible, consecutive windows sharing up to ``overlap`` tokens of whole pieces.

    ``keep_with_next``: character offsets (heading ends) a window must not end at, so a
    heading moves to the next window with the text it introduces."""
    pieces = _atomic(text, span, counter, max_tokens, tuple(separators))
    return _merge(text, pieces, counter, max_tokens, overlap, keep_with_next)


def _atomic(
    text: str, span: Span, counter: TokenCounter, max_tokens: int, seps: tuple[str, ...]
) -> list[Span]:
    """Pieces that each fit ``max_tokens``, using the first separator that splits the span."""
    start, end = span
    if counter.count(text[start:end]) <= max_tokens:
        return [span]
    for i, sep in enumerate(seps):
        parts, pos = [], start
        while (hit := text.find(sep, pos, end)) != -1:
            parts.append((pos, hit + len(sep)))  # separator stays with the left piece
            pos = hit + len(sep)
        if pos < end:
            parts.append((pos, end))
        if len(parts) > 1:
            out: list[Span] = []
            for part in parts:
                out.extend(_atomic(text, part, counter, max_tokens, seps[i + 1 :]))
            return out
    return _token_cut(text, span, counter, max_tokens)  # one huge "word" (e.g. a URL)


def _token_cut(text: str, span: Span, counter: TokenCounter, max_tokens: int) -> list[Span]:
    start, end = span
    offsets = counter.offsets(text[start:end])
    if not offsets:
        return [span]
    return [
        (start + offsets[i][0], start + offsets[min(i + max_tokens, len(offsets)) - 1][1])
        for i in range(0, len(offsets), max_tokens)
    ]


def _merge(
    text: str,
    pieces: list[Span],
    counter: TokenCounter,
    max_tokens: int,
    overlap: int,
    keep_with_next: frozenset[int] = frozenset(),
) -> list[Span]:
    counts = [counter.count(text[a:b]) for a, b in pieces]

    def ends_with_heading(end: int) -> bool:
        return len(text[:end].rstrip()) in keep_with_next

    windows: list[Span] = []
    i, n = 0, len(pieces)
    while i < n:
        j, total = i, 0
        while j < n and total + counts[j] <= max_tokens:
            total += counts[j]
            j += 1
        j = max(j, i + 1)
        # Piece counts are summed; a tokenizer can merge across a boundary, so re-check exactly.
        while j > i + 1 and counter.count(text[pieces[i][0] : pieces[j - 1][1]]) > max_tokens:
            j -= 1
        while j < n and j > i + 1 and ends_with_heading(pieces[j - 1][1]):
            j -= 1  # don't strand a heading at the end of a window
        if windows and pieces[j - 1][1] <= windows[-1][1]:
            # The overlap carried over pieces that no longer fit with anything new (seen:
            # a lone 5-token heading already inside the previous window). Drop the overlap.
            i = j
            continue
        windows.append((pieces[i][0], pieces[j - 1][1]))
        if j >= n:
            break
        k, carried = j, 0  # step back over whole pieces to create the overlap
        while k > i + 1 and carried + counts[k - 1] <= overlap:
            carried += counts[k - 1]
            k -= 1
        i = k
    return windows
