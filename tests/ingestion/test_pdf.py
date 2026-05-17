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


@pytest.mark.asyncio
async def test_pdf_loader_handles_corrupt_bytes(tmp_path: Path) -> None:
    """A corrupt PDF must either be rejected with a known pypdf/ValueError
    exception, or return an empty document list — but it must never silently
    succeed with garbage text, nor raise an unrelated exception type."""
    import pypdf.errors

    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4\n%this is not a valid pdf")

    loader = PdfLoader()
    try:
        docs = await loader.load(str(path))
    except (pypdf.errors.PyPdfError, pypdf.errors.PdfReadError, ValueError):
        # Acceptable: parser surfaced the corruption.
        return
    # Acceptable: loader returned empty (no documents produced).
    assert docs == [], (
        f"corrupt PDF unexpectedly produced {len(docs)} document(s); "
        "loader should either raise a known parser error or return []"
    )


@pytest.mark.asyncio
async def test_pdf_pipeline_isolates_corrupt_source(tmp_path: Path) -> None:
    """Per-source error isolation: one corrupt PDF must not crash the batch."""
    from gyroscope.ingestion.pipeline import IngestionPipeline

    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\n%this is not a valid pdf")

    pipeline = IngestionPipeline()
    docs = await pipeline.ingest([corrupt])
    # The pipeline catches loader exceptions and returns []. The contract is
    # "no crash" — an empty result is the canonical signal here.
    assert docs == [], f"expected [] when the only source is corrupt; got {docs!r}"
