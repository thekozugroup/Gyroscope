"""Local HTML file loader.

Strips ``<script>``/``<style>`` and other non-content tags via BeautifulSoup
and captures ``<title>`` when present.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from bs4 import BeautifulSoup

from gyroscope.core.logging import get_logger
from gyroscope.core.models import Document, DocumentKind
from gyroscope.ingestion.base import Loader, is_url

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")


def _clean_html_text(html: str) -> tuple[str, str | None, list[dict[str, int | str]]]:
    """Return ``(text, title, headings)`` extracted from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template", "iframe"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    headings: list[dict[str, int | str]] = []
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            txt = h.get_text(" ", strip=True)
            if txt:
                headings.append({"level": level, "text": txt})

    body = soup.body or soup
    text = body.get_text("\n", strip=True)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANKLINE_RE.sub("\n\n", text).strip() + "\n"
    if title is None:
        # Fall back to the first heading.
        for heading in headings:
            if heading["level"] == 1:
                title = str(heading["text"])
                break
    return text, title, headings


class HtmlLoader(Loader):
    """Loader for local ``.html`` / ``.htm`` files."""

    name = "html"
    extensions = (".html", ".htm")

    def can_load(self, source: str) -> bool:
        if is_url(source):
            return False
        return source.lower().endswith(self.extensions)

    async def load(self, source: str) -> list[Document]:
        path = Path(source)
        raw = await asyncio.to_thread(path.read_text, "utf-8", "ignore")
        text, title, headings = await asyncio.to_thread(_clean_html_text, raw)
        doc = Document(
            source=str(path.resolve()),
            kind=DocumentKind.HTML,
            text=text,
            title=title,
            metadata={
                "headings": headings,
                "size_bytes": len(text.encode("utf-8")),
            },
        )
        return [doc]
