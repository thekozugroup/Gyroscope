"""Tests for the plain text loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from gyroscope.core.models import DocumentKind
from gyroscope.ingestion.txt import TxtLoader


@pytest.mark.asyncio
async def test_txt_loader_basic(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_text("Title Line\nA paragraph with body text.\n", encoding="utf-8")

    loader = TxtLoader()
    assert loader.can_load(str(path))
    assert not loader.can_load("https://example.com")

    docs = await loader.load(str(path))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.kind == DocumentKind.TXT
    assert "paragraph with body text" in doc.text
    assert doc.title == "Title Line"
    assert doc.metadata["size_bytes"] > 0
    assert doc.source.endswith("hello.txt")
