"""Tests for the markdown loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from gyroscope.core.models import DocumentKind
from gyroscope.ingestion.markdown import MarkdownLoader


@pytest.mark.asyncio
async def test_markdown_loader_headings(tmp_path: Path) -> None:
    src = (
        "# Top Title\n"
        "\n"
        "Some intro.\n"
        "\n"
        "## Section A\n"
        "\n"
        "Content A.\n"
        "\n"
        "### Sub A\n"
        "\n"
        "Setext H2\n"
        "---------\n"
        "\n"
        "Body.\n"
        "\n"
        "```\n"
        "# not a heading\n"
        "```\n"
    )
    p = tmp_path / "doc.md"
    p.write_text(src, encoding="utf-8")

    loader = MarkdownLoader()
    assert loader.can_load(str(p))
    assert loader.can_load(str(tmp_path / "x.markdown"))
    assert not loader.can_load("foo.pdf")

    docs = await loader.load(str(p))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.kind == DocumentKind.MARKDOWN
    assert doc.title == "Top Title"
    headings = doc.metadata["headings"]
    levels = [h["level"] for h in headings]
    texts = [h["text"] for h in headings]
    assert 1 in levels and 2 in levels and 3 in levels
    assert "Top Title" in texts
    assert "Section A" in texts
    assert "Sub A" in texts
    assert "Setext H2" in texts
    # The fenced code block heading must NOT be picked up.
    assert "not a heading" not in texts
