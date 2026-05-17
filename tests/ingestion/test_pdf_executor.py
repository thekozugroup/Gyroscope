"""Tests for the PDF loader's dedicated thread pool.

The PDF loader uses a private ``_PDF_EXECUTOR`` ThreadPoolExecutor so a
flood of PDFs cannot starve the default ``asyncio.to_thread`` pool. Ops
can resize it at runtime via ``set_pdf_executor_workers``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gyroscope.core.models import DocumentKind
from gyroscope.ingestion import pdf as pdf_module
from gyroscope.ingestion.pdf import PdfLoader, set_pdf_executor_workers

reportlab = pytest.importorskip("reportlab", reason="reportlab is required to build PDF fixtures")
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402


def _make_minimal_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle("Executor Test Fixture")
    _width, height = LETTER
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 72, "Single page of executor-test content.")
    c.showPage()
    c.save()


def test_pdf_executor_is_a_thread_pool() -> None:
    """The module exposes a dedicated ThreadPoolExecutor for PDF parsing."""
    assert isinstance(pdf_module._PDF_EXECUTOR, ThreadPoolExecutor)
    # Default sizing rule: max(2, min(8, cpu_count or 4)).
    assert pdf_module._PDF_EXECUTOR._max_workers >= 2
    assert pdf_module._PDF_EXECUTOR._max_workers <= 8


def test_set_pdf_executor_workers_rebuilds_executor() -> None:
    """Resizing replaces the executor with a fresh, correctly-sized one."""
    original = pdf_module._PDF_EXECUTOR
    try:
        new_executor = set_pdf_executor_workers(2)
        assert new_executor is pdf_module._PDF_EXECUTOR
        assert new_executor is not original
        assert isinstance(new_executor, ThreadPoolExecutor)
        assert new_executor._max_workers == 2
    finally:
        # Restore the default to avoid leaking state across tests.
        set_pdf_executor_workers(original._max_workers)


def test_set_pdf_executor_workers_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        set_pdf_executor_workers(0)
    with pytest.raises(ValueError):
        set_pdf_executor_workers(-3)


@pytest.mark.asyncio
async def test_pdf_loader_works_after_executor_resize(tmp_path: Path) -> None:
    """After resizing the pool, ``PdfLoader.load`` still parses a real PDF."""
    path = tmp_path / "fixture.pdf"
    _make_minimal_pdf(path)

    original_workers = pdf_module._PDF_EXECUTOR._max_workers
    try:
        set_pdf_executor_workers(2)
        loader = PdfLoader()
        docs = await loader.load(str(path))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.kind == DocumentKind.PDF
        assert "executor-test content" in doc.text
    finally:
        set_pdf_executor_workers(original_workers)
