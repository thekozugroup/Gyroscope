"""Tests for the local HTML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from gyroscope.core.models import DocumentKind
from gyroscope.ingestion.html import HtmlLoader


@pytest.mark.asyncio
async def test_html_loader_strips_scripts(tmp_path: Path) -> None:
    html = (
        "<html><head><title>Greeting Page</title>"
        "<style>body{color:red}</style>"
        "</head><body>"
        "<h1>Greeting</h1>"
        "<p>Hello World</p>"
        "<script>alert('x')</script>"
        "<h2>Sub</h2><p>More text.</p>"
        "</body></html>"
    )
    path = tmp_path / "page.html"
    path.write_text(html, encoding="utf-8")

    loader = HtmlLoader()
    assert loader.can_load(str(path))
    assert not loader.can_load("https://example.com")

    docs = await loader.load(str(path))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.kind == DocumentKind.HTML
    assert doc.title == "Greeting Page"
    assert "Hello World" in doc.text
    assert "More text." in doc.text
    assert "alert" not in doc.text
    assert "color:red" not in doc.text
    headings = doc.metadata["headings"]
    levels = sorted(h["level"] for h in headings)
    assert levels == [1, 2]
