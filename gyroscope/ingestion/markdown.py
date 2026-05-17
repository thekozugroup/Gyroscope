"""Markdown loader.

Preserves headings as plain markdown text and emits a list of headings in
``metadata["headings"]`` so the curator can later reason about structure.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from gyroscope.core.logging import get_logger
from gyroscope.core.models import Document, DocumentKind
from gyroscope.ingestion.base import Loader, is_url

logger = get_logger(__name__)

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-+\s*$")


def _extract_headings(text: str) -> list[dict[str, int | str]]:
    """Return ATX/Setext headings as ``[{level, text}, ...]``."""
    headings: list[dict[str, int | str]] = []
    lines = text.splitlines()
    in_fence = False
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _ATX_HEADING_RE.match(raw)
        if m:
            level = len(m.group(1))
            headings.append({"level": level, "text": m.group(2).strip()})
            continue
        # Setext: underline applies to previous non-empty line.
        if i > 0 and (_SETEXT_H1_RE.match(stripped) or _SETEXT_H2_RE.match(stripped)):
            prev = lines[i - 1].strip()
            if prev and not _ATX_HEADING_RE.match(prev):
                level = 1 if _SETEXT_H1_RE.match(stripped) else 2
                headings.append({"level": level, "text": prev})
    return headings


def _minimal_cleanup(text: str) -> str:
    """Normalise line endings and trim trailing whitespace on each line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "\n".join(line.rstrip() for line in text.split("\n"))
    return cleaned.strip() + "\n"


class MarkdownLoader(Loader):
    """Loader for ``.md`` / ``.markdown`` files."""

    name = "markdown"
    extensions = (".md", ".markdown")

    def can_load(self, source: str) -> bool:
        if is_url(source):
            return False
        return source.lower().endswith(self.extensions)

    async def load(self, source: str) -> list[Document]:
        path = Path(source)
        raw = await asyncio.to_thread(path.read_text, "utf-8")
        text = _minimal_cleanup(raw)
        headings = _extract_headings(text)
        title: str | None = None
        for h in headings:
            if h["level"] == 1:
                title = str(h["text"])
                break
        if title is None and headings:
            title = str(headings[0]["text"])

        doc = Document(
            source=str(path.resolve()),
            kind=DocumentKind.MARKDOWN,
            text=text,
            title=title,
            metadata={
                "headings": headings,
                "size_bytes": len(text.encode("utf-8")),
            },
        )
        return [doc]
