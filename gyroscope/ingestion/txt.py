"""Plain text loader."""

from __future__ import annotations

import asyncio
from pathlib import Path

from gyroscope.core.logging import get_logger
from gyroscope.core.models import Document, DocumentKind
from gyroscope.ingestion.base import Loader, is_url

logger = get_logger(__name__)


class TxtLoader(Loader):
    """Trivial loader for ``.txt`` files."""

    name = "txt"
    extensions = (".txt",)

    def can_load(self, source: str) -> bool:
        if is_url(source):
            return False
        return source.lower().endswith(self.extensions)

    async def load(self, source: str) -> list[Document]:
        path = Path(source)
        text = await asyncio.to_thread(path.read_text, "utf-8")
        # The first non-empty line is treated as a best-effort title.
        title: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                title = stripped[:200]
                break
        doc = Document(
            source=str(path.resolve()),
            kind=DocumentKind.TXT,
            text=text,
            title=title,
            metadata={"size_bytes": len(text.encode("utf-8"))},
        )
        return [doc]
