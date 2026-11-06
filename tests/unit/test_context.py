"""Context assembly: units, block merging, budget fill, extractive compression (Phase 9c)."""

from __future__ import annotations

import numpy as np
import pytest

from omnirag.chunking.tokenizer import WhitespaceTokenizer
from omnirag.llm.chat import LLMError
from omnirag.retrieval.compression import GAP, Compressor, split_units
from omnirag.retrieval.context import (
    ContextAssembler,
    ContextOptions,
    build_blocks,
    fill,
)
from omnirag.retrieval.models import RetrievedChunk

COUNT = WhitespaceTokenizer().count
TABLE = "| Cell | Retention |\n| --- | --- |\n| AUR-C103 | 81.2% |\n| AUR-C105 | 89.6% |"


def _hit(cid: str, content: str, rank: int, doc: str = "d", section=("S",)) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id=doc, source_file=f"{doc}.md", rank=rank,
                          score=1.0, retriever="dense+bm25", content=content,
                          section_path=list(section), element_type="paragraph",
                          element_ids=[f"{cid}-e"], config_id="cfg")  # fmt: skip


def _row(cid: str, index: int, content: str, doc: str = "d") -> dict:
    return {"chunk_id": cid, "document_id": doc, "source_file": f"{doc}.md", "content": content,
            "chunk_index": index, "section_path": ["S"], "page_numbers": [index],
            "element_ids": [f"{cid}-e"], "origins": ["original"]}  # fmt: skip


# ---------------------------------------------------------------- units
def test_units_keep_abbreviations_and_table_headers() -> None:
    text = "Reviewed by Dr. Priya Raman. Approved on 2026-03-25.\n" + TABLE
    units = split_units(text)
    texts = [text[u.start : u.end] for u in units]
    assert texts[0] == "Reviewed by Dr. Priya Raman."  # no break after "Dr."
    assert texts[1] == "Approved on 2026-03-25."
    assert [u.header for u in units[2:]] == [True, True, False, False]
    assert {u.table for u in units[2:]} == {0}


def test_wrapped_sentences_and_list_numbers_stay_whole() -> None:
    """Measured in 9c: line-by-line splitting cut evidence quotes of the wrapped meeting notes."""
    text = (
        "2. Decision: move high-temperature testing to electrolyte E-7\n"
        "The team agreed to stop 45 C testing with E-5 and to qualify E-7 instead. "
        "Early E-7 cells\n"
        "(AUR-C105 and AUR-C106) are above 89% retention after 600 cycles at 45 C.\n"
        "- Marco: E-7 qualification report draft by 2026-03-26.\n"
        "id: M3\n"
        "name: Pilot line 100-cell batch\n"
        "Priya noted that milestone M2 (45 C stability demonstrated, due 2026-05-30) "
        "is at risk if\n"
        "E-7 swelling exceeds the 5% limit. A fallback is E-5 + FEC, which is at 83.1% after 800\n"
        "cycles on AUR-C104."
    )
    texts = [text[u.start : u.end] for u in split_units(text)]
    assert texts == [
        "2. Decision: move high-temperature testing to electrolyte E-7",  # heading, not "2."
        "The team agreed to stop 45 C testing with E-5 and to qualify E-7 instead.",
        "Early E-7 cells\n(AUR-C105 and AUR-C106) are above 89% retention "
        "after 600 cycles at 45 C.",
        "- Marco: E-7 qualification report draft by 2026-03-26.",  # list item: its own unit
        "id: M3",  # short record lines never join
        "name: Pilot line 100-cell batch",
        "Priya noted that milestone M2 (45 C stability demonstrated, due 2026-05-30) is at risk if"
        "\nE-7 swelling exceeds the 5% limit.",
        "A fallback is E-5 + FEC, which is at 83.1% after 800\ncycles on AUR-C104.",
    ]


# ---------------------------------------------------------------- blocks
def test_expansions_merge_in_reading_order_without_repeats() -> None:
    hits = [_hit("c3", "three", 1), _hit("x1", "other doc", 2, doc="x"), _hit("c5", "five", 3)]
    expansions = {
        "c3": [_row("c2", 2, "two"), _row("c3", 3, "three"), _row("c4", 4, "four")],
        "c5": [_row("c4", 4, "four"), _row("c5", 5, "five"), _row("c6", 6, "six")],
    }
    blocks = build_blocks(hits, expansions, COUNT)
    assert [b.chunk_ids for b in blocks] == [["c2", "c3", "c4", "c5", "c6"], ["x1"]]
    assert blocks[0].text == "two\n\nthree\n\nfour\n\nfive\n\nsix"  # c4 once
    assert blocks[0].hit_chunk_ids == ["c3", "c5"] and blocks[0].rank == 1
    assert blocks[0].tokens == COUNT("d.md > S\n" + blocks[0].text)


def test_fill_skips_a_block_that_does_not_fit_and_keeps_going() -> None:
    blocks = build_blocks([_hit("a", "w " * 5, 1), _hit("b", "w " * 50, 2),
                           _hit("c", "w " * 5, 3)], {}, COUNT)  # fmt: skip
    kept, skipped = fill(blocks, 20)
    assert [b.chunk_ids[0] for b in kept] == ["a", "c"] and skipped == ["b"]


# ---------------------------------------------------------------- compression
def test_lexical_compression_keeps_verbatim_units_and_table_headers() -> None:
    hits = [_hit("t", "Cycling results.\n" + TABLE, 1),
            _hit("p", "The weather was fine. AUR-C105 used E-7.", 2)]  # fmt: skip
    a = ContextAssembler(COUNT, compressor=Compressor())
    ctx = a.assemble("What retention did AUR-C105 reach?", hits,
                     ContextOptions(budget_tokens=64, compress="lexical"))  # fmt: skip
    table, para = ctx.blocks
    assert table.text == "| Cell | Retention |\n| --- | --- |" + GAP + "| AUR-C105 | 89.6% |"
    assert para.text == "AUR-C105 used E-7." and para.units_kept == 1 and para.units_total == 2
    assert ctx.tokens <= 64 and ctx.compression["method"] == "lexical"
    assert "| AUR-C105 | 89.6% |" in ctx.render()  # evidence survives word for word


def test_budget_is_respected_and_unmatched_units_dropped() -> None:
    hits = [_hit("a", "Alpha beta. Gamma delta.", 1), _hit("b", "nothing here.", 2)]
    a = ContextAssembler(COUNT, compressor=Compressor())
    ctx = a.assemble("alpha?", hits, ContextOptions(budget_tokens=64, compress="lexical"))
    assert [b.text for b in ctx.blocks] == ["Alpha beta."] and ctx.skipped == ["b"]


def test_llm_compression_keeps_chosen_numbers_and_falls_back() -> None:
    class LLM:
        def __init__(self, reply):
            self.reply, self.calls = reply, 0

        def chat(self, system, user, **kw):
            self.calls += 1
            if isinstance(self.reply, Exception):
                raise self.reply
            from omnirag.llm.chat import ChatResult

            return ChatResult("", self.reply, 1, 1, 1.0)

    hits = [_hit("a", "First fact. Second fact. Third fact.", 1)]
    llm = LLM({"keep": [3, 99, "x"]})
    a = ContextAssembler(COUNT, compressor=Compressor(llm=llm))
    for budget in (64, 128):  # one call serves both budgets
        ctx = a.assemble("Q?", hits, ContextOptions(budget_tokens=budget, compress="llm"))
        assert [b.text for b in ctx.blocks] == ["Third fact."]
    assert llm.calls == 1
    down = ContextAssembler(COUNT, compressor=Compressor(llm=LLM(LLMError("empty"))))
    ctx = down.assemble("first?", hits, ContextOptions(budget_tokens=64, compress="llm"))
    assert ctx.compression["fallback"].startswith("lexical") and ctx.blocks[0].text == "First fact."


def test_embedding_compression_ranks_by_cosine() -> None:
    class Embedder:
        def embed_documents(self, texts):
            return np.array([[1.0, 0.0] if "match" in t else [0.0, 1.0] for t in texts])

        def embed_query(self, text):
            return np.array([1.0, 0.0])

    other = "Other " * 29 + "words."
    found = "A match " + "x " * 27 + "here."
    hits = [_hit("a", f"{other} {found}", 1)]  # 2 units of ~30 tokens; 64 fits only one
    a = ContextAssembler(COUNT, compressor=Compressor(embedder=Embedder()))
    ctx = a.assemble("Q?", hits, ContextOptions(budget_tokens=64, compress="embedding"))
    assert ctx.blocks[0].text == found  # the higher cosine wins, although it comes second


def test_workflow_assembles_the_context_last() -> None:
    from workflow_fakes import FakeLLM, Retriever

    from omnirag.retrieval.transforms import QueryTransformer
    from omnirag.workflows.retrieval_graph import RetrievalWorkflow, Strategy

    dense = Retriever(
        {"Retention of AUR-C105?": ["a", "b"]},
        texts={"a": "Noise here. AUR-C105 reached 89.6%.", "b": "Unrelated."},
    )
    wf = RetrievalWorkflow(
        dense,
        Retriever({}),
        QueryTransformer(FakeLLM()),
        assembler=ContextAssembler(COUNT, compressor=Compressor()),
    )
    result = wf.retrieve("Retention of AUR-C105?", k=2, strategy=Strategy(
        grade=False, max_retries=0, compress="lexical", context_budget=64))  # fmt: skip
    wf.close()
    ctx = result.details["context"]
    assert result.details["trace"][-1]["node"] == "assemble"
    assert [b["text"] for b in ctx["blocks"]] == ["AUR-C105 reached 89.6%."]
    assert ctx["rendered"] == "[1] d.md\nAUR-C105 reached 89.6%."
    assert ctx["skipped"] == ["b"]


def test_missing_parts_are_explicit_errors() -> None:
    with pytest.raises(ValueError, match="graph store"):
        ContextAssembler(COUNT).assemble("q", [_hit("a", "x", 1)], ContextOptions(expand="section"))
    with pytest.raises(ValueError, match="compressor"):
        ContextAssembler(COUNT).assemble("q", [_hit("a", "x", 1)],
                                         ContextOptions(compress="lexical"))  # fmt: skip
