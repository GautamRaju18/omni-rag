"""Semantic chunking (fake embedder), LLM contextual enrichment (fake Ollama)."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from omnirag.chunking.enrichment import LLMContextualizer
from omnirag.chunking.models import ChunkingConfig, ChunkingStrategy
from omnirag.chunking.semantic import sentence_spans
from omnirag.chunking.strategies import chunk_document
from omnirag.chunking.tokenizer import WhitespaceTokenizer
from omnirag.ingestion.context import ExtractionContext
from omnirag.ingestion.models import DocumentRecord, ElementType, ExtractionMethod, FileKind

WS = WhitespaceTokenizer()
M = ExtractionMethod.TEXT_DECODE


class TopicEmbedder:
    """One-hot vectors by topic word: 'battery' sentences vs 'budget' sentences."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = np.array([t.count("battery"), t.count("budget"), 0.1], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def _record(ctx):
    return DocumentRecord(document_id=ctx.document_id, source_file="n.txt", file_kind=FileKind.TXT,
                          sha256="0" * 64, size_bytes=1)  # fmt: skip


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Prepared by Dr. Priya Raman. Reviewed.", ["Prepared by Dr. Priya Raman.", "Reviewed."]),
        ("3. Milestone M2 risk\nPriya noted it.", ["3. Milestone M2 risk", "Priya noted it."]),
        ("Cells failed at 45 °C! Next: E-7. Ok", ["Cells failed at 45 °C!", "Next: E-7.", "Ok"]),
        ("Retention was 89.6 % after 600 cycles.", ["Retention was 89.6 % after 600 cycles."]),
    ],
)  # fmt: skip
def test_sentence_spans(text: str, expected: list[str]) -> None:
    assert [text[a:b] for a, b in sentence_spans(text, 0, len(text))] == expected


def test_semantic_breaks_at_topic_shift_within_size() -> None:
    ctx = ExtractionContext("doc_00000000000000bb", "n.txt")
    battery = " ".join(f"The battery cell {i} cycled well." for i in range(4))
    budget = " ".join(f"The budget line {i} was approved." for i in range(4))
    ctx.add(ElementType.PARAGRAPH, f"{battery} {budget}", extraction_method=M)
    config = ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC, chunk_size=200,
                            breakpoint_percentile=80)  # fmt: skip
    embedder = TopicEmbedder()
    chunks = chunk_document(_record(ctx), ctx.elements, config, WS, embedder)
    assert [c.content for c in chunks] == [battery, budget]  # one break, at the topic change
    assert embedder.calls == 1  # all sentence windows in one batch
    assert all(c.partial_element_ids for c in chunks)  # each holds part of the paragraph
    assert config.config_id == "semantic-200-80"


def test_semantic_respects_size_limit() -> None:
    ctx = ExtractionContext("doc_00000000000000bb", "n.txt")
    ctx.add(ElementType.PARAGRAPH, " ".join(f"The battery cell {i} cycled." for i in range(30)),
            extraction_method=M)  # fmt: skip
    config = ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC, chunk_size=20)
    chunks = chunk_document(_record(ctx), ctx.elements, config, WS, TopicEmbedder())
    assert len(chunks) > 1 and all(c.token_count <= 20 for c in chunks)


def test_semantic_without_embedder_fails_clearly() -> None:
    ctx = ExtractionContext("doc_00000000000000bb", "n.txt")
    ctx.add(ElementType.PARAGRAPH, "Some text.", extraction_method=M)
    with pytest.raises(ValueError, match="needs an embedder"):
        chunk_document(_record(ctx), ctx.elements,
                       ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC), WS)  # fmt: skip


def test_llm_context_is_generated_and_only_for_embedding() -> None:
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompts.append(body["messages"][0]["content"])
        assert body["options"]["temperature"] == 0
        return httpx.Response(200, json={"message": {"content": " From the Q1 review:\n budget. "}})

    ctx = ExtractionContext("doc_00000000000000bb", "n.txt")
    for title, text in (("Budget", "Budget approved."), ("Cells", "Cells cycled.")):
        ctx.add(ElementType.HEADING, title, extraction_method=M, metadata={"level": 1})
        ctx.add(ElementType.PARAGRAPH, text, extraction_method=M, section_path=[title])
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE, chunk_size=16,
                            contextual_llm=True)  # fmt: skip
    chunks = chunk_document(_record(ctx), ctx.elements, config, WS)
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    contextualizer = LLMContextualizer("http://ollama", "chat:test", WS, client=client)
    contextualizer.enrich(chunks, ctx.elements)
    assert config.config_id == "structure-16-llmctx" and len(chunks) == 2
    assert chunks[0].content == "Budget\n\nBudget approved."  # citations show the source text
    assert chunks[0].context == "From the Q1 review: budget."
    assert chunks[0].embedding_text().startswith("From the Q1 review")
    assert chunks[0].metadata["context_origin"] == "generated"
    assert chunks[0].metadata["context_model"] == "chat:test"
    assert all("Budget approved." in p and "Cells cycled." in p for p in prompts)  # whole doc
    assert contextualizer.stats["calls"] == len(chunks)
