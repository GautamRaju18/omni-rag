"""Token counting with the embedding model's own tokenizer.

Chunk sizes are in **bge-m3 tokens**, because the embedding model's limit is what matters. Measured
2026-11-04: the Hugging Face ``tokenizer.json`` (pinned revision) gives exactly Ollama's
``prompt_eval_count`` minus the 2 special tokens, on 5 of 5 test strings (prose, a table row,
German text, a Markdown table and ``Na₂Fe[Fe(CN)₆]``). "4 characters per token" would be wrong
here: prose is 3.2 chars/token, table rows 2.1 and the formula 1.4.

The tokenizer file (17 MB) is downloaded once from a URL built from settings (never from
document content), verified against a pinned SHA-256, and stored in ``<data_dir>/models``.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Protocol

import httpx

Span = tuple[int, int]


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def offsets(self, text: str) -> list[Span]:
        """Character span of every token (no special tokens)."""
        ...


class HFTokenizer:
    def __init__(self, path: Path, name: str) -> None:
        from tokenizers import Tokenizer  # Rust, fast and thread-safe for encode()

        self._tok = Tokenizer.from_file(str(path))
        self.name = name

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids) if text else 0

    def offsets(self, text: str) -> list[Span]:
        return list(self._tok.encode(text, add_special_tokens=False).offsets) if text else []


class WhitespaceTokenizer:
    """Deterministic stand-in for unit tests: one token per run of non-space characters."""

    name = "whitespace"
    _TOKEN = re.compile(r"\S+")

    def count(self, text: str) -> int:
        return len(self._TOKEN.findall(text))

    def offsets(self, text: str) -> list[Span]:
        return [m.span() for m in self._TOKEN.finditer(text)]


class TokenizerDownloadError(RuntimeError):
    pass


def ensure_tokenizer(
    models_dir: Path,
    repo: str,
    revision: str,
    sha256: str,
    *,
    timeout_s: float = 120.0,
    client: httpx.Client | None = None,
) -> Path:
    """Return the local path of ``tokenizer.json``, downloading and verifying it if needed."""
    target = models_dir / f"{repo.replace('/', '--')}-{revision[:12]}" / "tokenizer.json"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/tokenizer.json"
    digest = hashlib.sha256()
    http = client or httpx.Client(follow_redirects=True, timeout=timeout_s)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        try:
            with http.stream("GET", url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1 << 20):
                    digest.update(chunk)
                    tmp.write(chunk)
        except httpx.HTTPError as exc:
            Path(tmp.name).unlink(missing_ok=True)
            raise TokenizerDownloadError(f"Could not download {url}: {exc}") from exc
        finally:
            if client is None:
                http.close()
    if digest.hexdigest() != sha256:
        Path(tmp.name).unlink(missing_ok=True)
        raise TokenizerDownloadError(f"Checksum mismatch for {url}: got {digest.hexdigest()}")
    Path(tmp.name).replace(target)
    return target
