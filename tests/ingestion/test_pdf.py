"""Tests for the PDF loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from gyroscope.core.models import DocumentKind
from gyroscope.ingestion.pdf import PdfLoader

reportlab = pytest.importorskip("reportlab", reason="reportlab is required to build PDF fixtures")
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402


def _make_pdf(
    path: Path,
    pages: list[list[str]],
    header: str | None = None,
    footer: str | None = None,
    title: str | None = None,
) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    if title:
        c.setTitle(title)
    _width, height = LETTER
    for body_lines in pages:
        y = height - 72  # 1 inch margin
        if header:
            c.setFont("Helvetica", 9)
            c.drawString(72, height - 36, header)
        c.setFont("Helvetica", 12)
        for line in body_lines:
            c.drawString(72, y, line)
            y -= 16
        if footer:
            c.setFont("Helvetica", 9)
            c.drawString(72, 36, footer)
        c.showPage()
    c.save()


@pytest.mark.asyncio
async def test_pdf_loader_extracts_pages_and_strips_boilerplate(tmp_path: Path) -> None:
    path = tmp_path / "fixture.pdf"
    pages = [
        ["Quantum Surveying Handbook", "Introduction to the topic."],
        ["Chapter Two", "Discussion of measurement methods."],
        ["Chapter Three", "Cost planning fundamentals."],
        ["Chapter Four", "Risk and contingency."],
    ]
    _make_pdf(
        path,
        pages,
        header="Quantum Surveying Handbook (c) 2024",
        footer="Confidential",
        title="Quantum Surveying Handbook",
    )

    loader = PdfLoader()
    assert loader.can_load(str(path))

    docs = await loader.load(str(path))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.kind == DocumentKind.PDF
    assert doc.title is not None
    assert "Quantum" in doc.title
    assert doc.metadata["page_count"] == 4
    page_map = doc.metadata["page_map"]
    assert len(page_map) == 4
    assert page_map[0]["page"] == 1
    assert page_map[-1]["page"] == 4
    # Offsets are non-overlapping and monotonically increasing.
    from itertools import pairwise

    for prev, nxt in pairwise(page_map):
        assert nxt["start"] >= prev["end"]
    # Repeating header / footer should have been stripped.
    assert "Confidential" not in doc.text
    # Each chapter heading is present.
    for chapter in ("Chapter Two", "Chapter Three", "Chapter Four"):
        assert chapter in doc.text
    # The header text used to appear on every page; after stripping it
    # should NOT appear inside the body more than once at most.
    assert doc.text.count("Quantum Surveying Handbook (c) 2024") <= 1


@pytest.mark.asyncio
async def test_pdf_loader_does_not_load_urls(tmp_path: Path) -> None:
    loader = PdfLoader()
    assert not loader.can_load("https://example.com/foo.pdf")
