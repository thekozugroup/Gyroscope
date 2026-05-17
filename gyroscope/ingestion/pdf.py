"""PDF loader using pypdf with pdfplumber fallback for tricky pages.

Per-page character offsets are recorded in ``metadata["page_map"]`` as
``[{"page": 1, "start": 0, "end": 1234}, ...]`` so downstream chunkers can
translate a slice of ``text`` back to a page number for citation.

A best-effort title is detected from (in order):
1. PDF document outline / metadata ``Title`` field.
2. The largest / first non-empty heading-like line on page 1.

Repeating running headers / footers are stripped: any short line that appears
on a majority of pages at the top or bottom is treated as boilerplate and
removed from the per-page text.
"""

from __future__ import annotations

import asyncio
import atexit
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from gyroscope.core.logging import get_logger
from gyroscope.core.models import Document, DocumentKind
from gyroscope.ingestion.base import Loader, is_url

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dedicated thread pool for PDF parsing.
#
# Without this, ``asyncio.to_thread`` would queue PDF parses onto the default
# loop executor (``min(32, os.cpu_count()+4)`` workers shared with every
# other ``to_thread`` caller in the process). Ingesting a directory of many
# PDFs would then starve unrelated background work. A small, dedicated pool
# isolates PDF work and lets ops tune the parallelism explicitly.
# ---------------------------------------------------------------------------


def _default_pdf_workers() -> int:
    return max(2, min(8, os.cpu_count() or 4))


_PDF_EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=_default_pdf_workers(),
    thread_name_prefix="gyroscope-pdf",
)
atexit.register(_PDF_EXECUTOR.shutdown, wait=False)


def set_pdf_executor_workers(n: int) -> ThreadPoolExecutor:
    """Resize the dedicated PDF parsing thread pool.

    Rebuilds ``_PDF_EXECUTOR`` with the requested worker count, shutting the
    previous executor down (without waiting) so currently-running parses
    complete on the old workers. Returns the new executor.
    """
    if n <= 0:
        raise ValueError("PDF executor worker count must be positive.")
    global _PDF_EXECUTOR
    old = _PDF_EXECUTOR
    _PDF_EXECUTOR = ThreadPoolExecutor(
        max_workers=n,
        thread_name_prefix="gyroscope-pdf",
    )
    atexit.register(_PDF_EXECUTOR.shutdown, wait=False)
    old.shutdown(wait=False)
    return _PDF_EXECUTOR


# ---------------------------------------------------------------------------
# Header / footer detection
# ---------------------------------------------------------------------------


_HEADER_FOOTER_LINES = 3  # lines per page to inspect at top and bottom
_REPEAT_FRACTION = 0.6  # appears on >= 60 % of pages → boilerplate


def _candidate_lines(page_text: str) -> tuple[list[str], list[str]]:
    """Return ``(top_lines, bottom_lines)`` candidates for header/footer."""
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    top = lines[:_HEADER_FOOTER_LINES]
    bot = lines[-_HEADER_FOOTER_LINES:] if len(lines) >= _HEADER_FOOTER_LINES else []
    return top, bot


def _detect_boilerplate(pages_text: list[str]) -> set[str]:
    """Detect lines that repeat at top/bottom across many pages."""
    if len(pages_text) < 3:
        return set()
    counter: Counter[str] = Counter()
    for ptxt in pages_text:
        top, bot = _candidate_lines(ptxt)
        seen: set[str] = set()
        for ln in (*top, *bot):
            # Ignore long lines (likely real content) and pure numbers
            # (page numbers vary anyway).
            if len(ln) > 120:
                continue
            if ln in seen:
                continue
            seen.add(ln)
            counter[ln] += 1
    threshold = max(2, int(len(pages_text) * _REPEAT_FRACTION))
    return {ln for ln, n in counter.items() if n >= threshold}


def _strip_boilerplate(page_text: str, boilerplate: set[str]) -> str:
    if not boilerplate:
        return page_text
    kept: list[str] = []
    for ln in page_text.splitlines():
        if ln.strip() in boilerplate:
            continue
        kept.append(ln)
    # Collapse leading / trailing blank lines that boilerplate removal left.
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_page_text(page: Any, page_idx: int) -> str:
    """Extract text from a pypdf page, falling back to pdfplumber on failure."""
    try:
        text = page.extract_text() or ""
    except Exception as exc:
        logger.debug("pypdf failed on page %d: %s", page_idx + 1, exc)
        text = ""
    if text.strip():
        return text
    # Fallback to pdfplumber.
    try:
        import pdfplumber  # local import: heavy dep
    except Exception:
        return ""
    try:
        reader_path = getattr(page, "_reader", None)
        stream = getattr(reader_path, "stream", None)
        if stream is None:
            return ""
        stream.seek(0)
        with pdfplumber.open(stream) as pdf:
            if page_idx < len(pdf.pages):
                return pdf.pages[page_idx].extract_text() or ""
    except Exception as exc:
        logger.debug("pdfplumber fallback failed on page %d: %s", page_idx + 1, exc)
    return ""


def _detect_title_from_outline(reader: PdfReader) -> str | None:
    """Try to read the title from the document info dict."""
    try:
        meta = reader.metadata
    except Exception:
        return None
    if meta is None:
        return None
    title = getattr(meta, "title", None)
    if title and str(title).strip():
        return str(title).strip()
    return None


def _detect_title_from_first_page(first_page_text: str) -> str | None:
    """Heuristic: take the first non-trivial line of page 1."""
    for raw in first_page_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) < 4:
            continue
        # Skip lines that look like page numbers or dates only.
        if line.replace(" ", "").isdigit():
            continue
        return line[:200]
    return None


def _load_pdf_sync(path: Path) -> Document:
    """Synchronous heavy lifting; called via ``asyncio.to_thread``."""
    reader = PdfReader(str(path))
    raw_pages: list[str] = []
    for i, page in enumerate(reader.pages):
        raw_pages.append(_extract_page_text(page, i))

    boilerplate = _detect_boilerplate(raw_pages)
    cleaned_pages = [_strip_boilerplate(p, boilerplate) for p in raw_pages]

    # Build the concatenated text and per-page offsets in one pass.
    text_parts: list[str] = []
    page_map: list[dict[str, int]] = []
    cursor = 0
    for i, ptxt in enumerate(cleaned_pages):
        start = cursor
        if ptxt:
            text_parts.append(ptxt)
            cursor += len(ptxt)
        # Page separator (kept inside the offset for that page so chunkers
        # never land "between" pages).
        if i < len(cleaned_pages) - 1:
            sep = "\n\n"
            text_parts.append(sep)
            cursor += len(sep)
        end = cursor
        page_map.append({"page": i + 1, "start": start, "end": end})

    full_text = "".join(text_parts).strip() + ("\n" if text_parts else "")

    title = _detect_title_from_outline(reader)
    if not title and raw_pages:
        title = _detect_title_from_first_page(raw_pages[0])

    author: str | None = None
    try:
        meta = reader.metadata
        if meta is not None:
            author_val = getattr(meta, "author", None)
            if author_val:
                author = str(author_val).strip() or None
    except Exception:
        author = None

    metadata: dict[str, Any] = {
        "page_count": len(raw_pages),
        "page_map": page_map,
        "removed_boilerplate": sorted(boilerplate),
    }
    if author:
        metadata["author"] = author

    return Document(
        source=str(path.resolve()),
        kind=DocumentKind.PDF,
        text=full_text,
        title=title,
        metadata=metadata,
    )


class PdfLoader(Loader):
    """Loader for ``.pdf`` files."""

    name = "pdf"
    extensions = (".pdf",)

    def can_load(self, source: str) -> bool:
        if is_url(source):
            return False
        return source.lower().endswith(self.extensions)

    async def load(self, source: str) -> list[Document]:
        path = Path(source)
        loop = asyncio.get_running_loop()
        doc = await loop.run_in_executor(_PDF_EXECUTOR, _load_pdf_sync, path)
        return [doc]
