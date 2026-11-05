"""Exact in-memory retrievers for benchmarking (the Neo4j indexes come in Phases 5–6).

* ``DenseIndex``: cosine similarity over L2-normalised vectors, exact (no ANN approximation),
  so differences between chunking strategies aren't hidden by index recall.
* ``BM25Index``: Okapi BM25 (k1=1.5, b=0.75). The tokenizer keeps identifiers whole:
  ``AUR-C103``, ``E-7``, ``2026-03-19`` and ``89.6`` are single terms, which is what makes
  lexical search strong on exact-keyword questions.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TERM = re.compile(r"[0-9a-zà-ÿ]+(?:[-./][0-9a-zà-ÿ]+)*")


def terms(text: str) -> list[str]:
    return _TERM.findall(text.lower())


class DenseIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, query: np.ndarray, k: int) -> list[int]:
        scores = self.vectors @ query
        return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(terms(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[int]:
        q = terms(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                    s += self.idf[t] * tf * (self.k1 + 1) / norm
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return order[:k]
