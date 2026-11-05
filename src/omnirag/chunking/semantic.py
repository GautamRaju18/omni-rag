"""Semantic chunking: put boundaries where the topic shifts.

1. Split each section's prose into sentences (element boundaries are sentence boundaries too).
2. Embed every sentence *with its neighbours* (a window of 3). Single short sentences give noisy
   embeddings; the window smooths that (Greg Kamradt's "buffer").
3. Distance between consecutive windows = 1 − cosine. Break where the distance is above the
   ``breakpoint_percentile`` of all distances in the document: "unusually different from its
   neighbour" relative to this document, not an absolute threshold.
4. Constraints: never cross a section, never exceed ``chunk_size``, never break right after a
   heading. Tables and figures are standalone, exactly as in the multimodal strategy.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from omnirag.chunking.text import Span

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
_ABBREVIATIONS = ("Dr.", "Mr.", "Ms.", "Mrs.", "e.g.", "i.e.", "vs.", "etc.", "Fig.", "No.")
# "3. Milestone M2 risk": a list/section number at the start of a line is not a sentence end.
# (Seen in the meeting notes: it produced a 1-token chunk "3.")
_LIST_MARKER = re.compile(r"(?:^|\n)\s*(?:\d{1,3}|[a-zA-Z])[.)]$")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def sentence_spans(text: str, start: int, end: int) -> list[Span]:
    """Sentence spans inside ``text[start:end]`` (newlines always end a sentence)."""
    spans, pos = [], start
    for match in _SENTENCE_END.finditer(text, start, end):
        if match.group().startswith((" ", "\t")):
            before = text[max(pos, match.start() - 8) : match.start()]
            if before.endswith(_ABBREVIATIONS):
                continue  # "Dr. Priya Raman" is one sentence
            if _LIST_MARKER.search(text[pos : match.start()]):
                continue  # "3. Milestone ..." is one sentence
        if text[pos : match.start()].strip():
            spans.append((pos, match.start()))
        pos = match.end()
    if text[pos:end].strip():
        spans.append((pos, end))
    return spans


def window_texts(text: str, sentences: list[Span], buffer: int = 1) -> list[str]:
    return [
        text[sentences[max(0, i - buffer)][0] : sentences[min(len(sentences) - 1, i + buffer)][1]]
        for i in range(len(sentences))
    ]


def distances(vectors: np.ndarray) -> np.ndarray:
    """1 − cosine between consecutive (L2-normalised) vectors."""
    if len(vectors) < 2:
        return np.zeros(0, dtype=np.float32)
    return 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
