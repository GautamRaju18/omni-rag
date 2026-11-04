"""Per-loader behaviour on small, hand-written inputs."""

from __future__ import annotations

import json

import pytest

from omnirag.ingestion.loaders.csv_loader import CsvLoader
from omnirag.ingestion.loaders.html_loader import HtmlLoader
from omnirag.ingestion.loaders.json_loader import JsonLoader
from omnirag.ingestion.loaders.markdown_loader import MarkdownLoader
from omnirag.ingestion.loaders.text_loader import TextLoader
from omnirag.ingestion.models import ElementType


def _types(ctx) -> list[str]:
    return [e.element_type.value for e in ctx.elements]


# ---------------------------------------------------------------- text
def test_text_paragraphs_and_line_numbers(run_loader) -> None:
    ctx, meta = run_loader(TextLoader(), "n.txt", "Title line\nsecond line\n\n\nPara two\n")
    assert [e.content for e in ctx.elements] == ["Title line\nsecond line", "Para two"]
    assert ctx.elements[0].metadata == {"line_start": 1, "line_end": 2}
    assert ctx.elements[1].metadata == {"line_start": 5, "line_end": 5}
    assert meta["encoding"] == "utf-8-sig"


def test_text_cp1252_fallback_lowers_confidence(run_loader) -> None:
    # cp1252 bytes 0xE9 (é) and 0x80 (€) are invalid UTF-8, forcing the fallback.
    ctx, meta = run_loader(TextLoader(), "legacy.txt", "Café € price".encode("cp1252"))
    assert ctx.elements[0].content == "Café € price"
    assert ctx.elements[0].confidence == 0.8
    assert meta["encoding"] == "cp1252"
    assert ctx.warnings


# ------------------------------------------------------------ markdown
MD = """---
owner: test
---
# Guide

Intro *text*
continues here.

## Setup

- one
- two

```python
    indented = True
```

| Key | Value |
|-----|-------|
| a   | 1     |

# Second
![diagram](d.png)
"""


def test_markdown_structure(run_loader) -> None:
    ctx, meta = run_loader(MarkdownLoader(), "g.md", MD)
    assert _types(ctx) == [
        "heading", "paragraph", "heading", "list", "code", "table", "table_row", "heading", "image",
    ]  # fmt: skip
    by_type = {e.element_type: e for e in ctx.elements}
    assert by_type[ElementType.PARAGRAPH].content == "Intro *text* continues here."
    assert by_type[ElementType.CODE].content == "    indented = True"
    assert by_type[ElementType.CODE].metadata["language"] == "python"
    assert by_type[ElementType.TABLE_ROW].content == "Key: a | Value: 1"
    assert by_type[ElementType.LIST].section_path == ["Guide", "Setup"]
    # "# Second" resets the trail: a new H1 is not nested under "Guide".
    assert by_type[ElementType.IMAGE].section_path == ["Second"]
    assert by_type[ElementType.IMAGE].metadata["src"] == "d.png"
    assert meta["title"] == "Guide"
    assert meta["front_matter"] == "owner: test"
    # Line numbers survive front-matter removal: "# Guide" is line 4 of the file.
    assert ctx.elements[0].metadata["line_start"] == 4


def test_markdown_hash_inside_code_is_not_a_heading(run_loader) -> None:
    ctx, _ = run_loader(MarkdownLoader(), "c.md", "```bash\n# not a heading\n```\n")
    assert _types(ctx) == ["code"]


# ---------------------------------------------------------------- html
HTML = """<html lang="de"><head><title>T</title><script>evil()</script>
<meta name="description" content="desc"></head>
<body><nav>Home | About</nav>
<main><h1>Main</h1><div>Loose <b>bold</b> text<p>Para
  one</p></div>
<table><tr><td>k</td><td>v</td></tr><tr><td>x</td><td>1</td></tr></table>
<ol><li>first<ul><li>nested</li></ul></li><li>second</li></ol>
<pre>  code  block</pre></main>
<footer>Copyright</footer></body></html>"""


def test_html_main_content_only(run_loader) -> None:
    ctx, meta = run_loader(HtmlLoader(), "p.html", HTML)
    text = " ".join(e.content for e in ctx.elements)
    for noise in ("evil", "Home", "Copyright"):
        assert noise not in text
    assert _types(ctx) == [
        "heading",
        "paragraph",
        "paragraph",
        "table",
        "table_row",
        "list",
        "code",
    ]
    assert ctx.elements[1].content == "Loose bold text"
    assert ctx.elements[2].content == "Para one"  # newline inside <p> collapsed
    assert ctx.elements[5].content == "1. first\n  - nested\n2. second"
    assert ctx.elements[6].content == "  code  block"  # <pre> keeps spacing
    assert meta["title"] == "T" and meta["description"] == "desc" and meta["language"] == "de"
    assert meta["content_root"] == "main"


def test_html_table_without_th_has_lower_confidence(run_loader) -> None:
    ctx, _ = run_loader(HtmlLoader(), "p.html", HTML)
    table = next(e for e in ctx.elements if e.element_type is ElementType.TABLE)
    assert table.metadata["header_inferred"] is True
    assert table.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------- json
def test_json_records_and_paths(run_loader) -> None:
    data = {"name": "A", "tags": ["x", "y"], "team": [{"n": "P", "skills": {"py": 5}}]}
    ctx, meta = run_loader(JsonLoader(), "d.json", json.dumps(data))
    assert [e.content for e in ctx.elements] == ["name: A\ntags: x, y", "n: P", "py: 5"]
    paths = [e.metadata["json_path"] for e in ctx.elements]
    assert paths == ["$", "$.team[0]", "$.team[0].skills"]
    root, member, skills = ctx.elements
    assert member.parent_element_id == root.element_id
    assert skills.parent_element_id == member.element_id
    assert member.section_path == ["team[0]"]
    assert meta["top_level_type"] == "dict"


def test_json_invalid_raises(run_loader) -> None:
    with pytest.raises(json.JSONDecodeError):
        run_loader(JsonLoader(), "bad.json", "{not json")


# ----------------------------------------------------------------- csv
def test_csv_semicolon_dialect_with_header(run_loader) -> None:
    ctx, meta = run_loader(CsvLoader(), "s.csv", "id;value;note\nA1;10;ok\nA2;20;\n")
    assert meta["delimiter"] == ";" and meta["header_detected"] is True
    assert meta["columns"] == ["id", "value", "note"]
    table, row1, row2 = ctx.elements
    assert row1.content == "id: A1 | value: 10 | note: ok"
    assert row2.content == "id: A2 | value: 20"  # empty cell omitted
    assert row1.parent_element_id == table.element_id


def test_csv_without_header_gets_generated_columns(run_loader) -> None:
    ctx, meta = run_loader(CsvLoader(), "n.csv", "1,2.5,3\n4,5.5,6\n7,8.5,9\n")
    assert meta["header_detected"] is False
    assert ctx.elements[1].content == "column_1: 1 | column_2: 2.5 | column_3: 3"
