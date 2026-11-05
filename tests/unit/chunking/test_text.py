"""Span-based splitting primitives (whitespace tokens: one token per word)."""

from __future__ import annotations

import pytest

from omnirag.chunking.text import Assembled, Segment, split_spans, strip_span, token_windows
from omnirag.chunking.tokenizer import WhitespaceTokenizer

WS = WhitespaceTokenizer()


def _texts(text: str, spans) -> list[str]:
    return [text[a:b].strip() for a, b in spans]


def test_recursive_split_prefers_paragraphs_then_sentences() -> None:
    # Paragraph 2 (8 words) is too big, so it is split at sentence ends; small neighbouring
    # pieces are then packed up to the limit, but a window never ends mid-sentence.
    text = "one two three four.\n\nfive six seven. eight nine ten. eleven twelve."
    spans = split_spans(text, (0, len(text)), WS, max_tokens=6)
    assert _texts(text, spans) == [
        "one two three four.",
        "five six seven. eight nine ten.",
        "eleven twelve.",
    ]
    assert all(WS.count(text[a:b]) <= 6 for a, b in spans)


def test_overlap_repeats_whole_trailing_pieces() -> None:
    text = "a b. c d. e f. g h."
    spans = split_spans(text, (0, len(text)), WS, max_tokens=4, overlap=2)
    assert _texts(text, spans) == ["a b. c d.", "c d. e f.", "e f. g h."]


def test_unbreakable_word_run_is_cut_by_tokens() -> None:
    text = " ".join(f"w{i}" for i in range(10))
    spans = split_spans(text, (0, len(text)), WS, max_tokens=4, separators=())
    assert [WS.count(text[a:b]) for a, b in spans] == [4, 4, 2]


def test_heading_is_not_stranded_at_window_end() -> None:
    text = "intro words here\n\nHeading\n\nbody one two three"
    heading_end = text.index("Heading") + len("Heading")
    spans = split_spans(text, (0, len(text)), WS, max_tokens=5,
                        keep_with_next=frozenset({heading_end}))  # fmt: skip
    assert _texts(text, spans) == ["intro words here", "Heading\n\nbody one two three"]


def test_token_windows_exact_size_and_step() -> None:
    text = " ".join(str(i) for i in range(10))
    spans = token_windows(text, WS, size=4, overlap=1)
    assert _texts(text, spans) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


class Overcounting(WhitespaceTokenizer):
    """Mimics sub-word tokenizers: a slice that starts mid-'word' costs one extra token."""

    def count(self, text: str) -> int:
        return super().count(text) + (1 if text and text[0].islower() else 0)


def test_token_windows_trim_when_retokenized_slice_is_longer() -> None:
    text = "A b c d E f g h"
    spans = token_windows(text, Overcounting(), size=4, overlap=0)
    assert all(Overcounting().count(text[a:b]) <= 4 for a, b in spans)


def test_overlapping_marks_partial_and_zero_length_segments() -> None:
    out = Assembled("")
    seg = lambda eid: Segment(0, 0, eid, "paragraph", 1, [], "original", 1.0)  # noqa: E731
    out.add("", seg("img"))  # an image with no text sits at offset 0
    out.add("alpha beta", seg("p1"))
    out.add("gamma delta", seg("p2"))
    hits = {s.element_id: partial for s, partial in out.overlapping((0, 16))}
    assert hits == {"img": False, "p1": False, "p2": True}


@pytest.mark.parametrize(
    ("raw", "expected"), [("  ab \n", (2, 4)), ("ab", (0, 2)), ("   ", (3, 3))]
)
def test_strip_span(raw: str, expected) -> None:
    assert strip_span(raw, (0, len(raw))) == expected
