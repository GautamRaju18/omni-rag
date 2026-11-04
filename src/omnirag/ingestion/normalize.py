"""Text normalization: the same text must look the same regardless of source format.

Without this, ``"ﬁle"`` (a PDF ligature) never matches a BM25 query for ``"file"``, a non-breaking
space splits tokens differently, and invisible zero-width characters change embeddings.

We use Unicode **NFC** plus targeted fixes instead of NFKC: NFKC would also rewrite ``x²`` to
``x2`` and ``½`` to ``1⁄2``, silently changing technical meaning.
"""

from __future__ import annotations

import re
import unicodedata

_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st"})
_SPACES = str.maketrans({" ": " ", " ": " ", " ": " ", "　": " "})
_INVISIBLE = re.compile("[​‌‍⁠﻿­]")  # zero-width, BOM, soft hyphen
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # keeps \t \n \r
# Vertical tab is PowerPoint's soft line break (python-pptx returns it as "\v"); form feed is a
# page break in plain text and PDFs. Both separate words, so they become newlines instead of
# being deleted with the other control characters ("line\vbreak" must not become "linebreak").
_LINE_SEPARATORS = str.maketrans({"\v": "\n", "\f": "\n", " ": "\n", " ": "\n"})
_HSPACE_RUN = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str, *, preserve_layout: bool = False) -> str:
    """Normalize unicode and whitespace.

    ``preserve_layout=True`` keeps indentation and inner spacing (code blocks, Markdown tables)
    and only strips trailing spaces per line.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_LINE_SEPARATORS)
    text = text.translate(_LIGATURES).translate(_SPACES)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    if preserve_layout:
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return _BLANK_LINES.sub("\n\n", text).strip("\n")  # keep first-line indentation
    text = "\n".join(_HSPACE_RUN.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()
