"""IngestionPipeline — top-level entry point for Phase 1.

Glob-expands filesystem inputs (including directory recursion), routes each
resolved source through the appropriate loader, runs loads concurrently with
a bounded semaphore, isolates failures so one bad source cannot kill the
batch, and returns a deterministically ordered ``list[Document]``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

from gyroscope.core.logging import get_logger
from gyroscope.core.models import Document
from gyroscope.ingestion.base import Loader, LoaderRegistry, is_url
from gyroscope.ingestion.docx import DocxLoader
from gyroscope.ingestion.html import HtmlLoader
from gyroscope.ingestion.markdown import MarkdownLoader
from gyroscope.ingestion.pdf import PdfLoader
from gyroscope.ingestion.txt import TxtLoader
from gyroscope.ingestion.web import WebLoader

logger = get_logger(__name__)


_DEFAULT_FILE_EXTS: tuple[str, ...] = (
    ".pdf",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".docx",
    ".txt",
)


def _default_registry(web_loader: WebLoader | None = None) -> LoaderRegistry:
    reg = LoaderRegistry()
    reg.register(PdfLoader())
    reg.register(MarkdownLoader())
    reg.register(HtmlLoader())
    reg.register(DocxLoader())
    reg.register(TxtLoader())
    reg.register(web_loader or WebLoader())
    return reg


def _expand_path(path: Path) -> list[str]:
    """Expand a file / directory / glob into a flat list of file paths."""
    s = str(path)
    if any(ch in s for ch in "*?[]"):
        return [str(Path(m).resolve()) for m in sorted(Path().glob(s))]
    if path.is_dir():
        files = [
            str(p.resolve())
            for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() in _DEFAULT_FILE_EXTS
        ]
        return files
    if path.exists() and path.is_file():
        return [str(path.resolve())]
    # Path does not exist — surface as-is so the loader emits a sensible error.
    return [str(path)]


def _expand_sources(sources: Iterable[str | Path]) -> list[str]:
    """Apply glob / directory expansion and de-duplicate while keeping order."""
    expanded: list[str] = []
    for s in sources:
        if isinstance(s, str) and is_url(s):
            expanded.append(s)
            continue
        path = Path(s)
        expanded.extend(_expand_path(path))
    seen: set[str] = set()
    deduped: list[str] = []
    for s in expanded:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped


class IngestionPipeline:
    """Resolve sources via a registry and emit a deterministic Document list."""

    def __init__(
        self,
        registry: LoaderRegistry | None = None,
        *,
        max_concurrent: int = 8,
        web_loader: WebLoader | None = None,
    ) -> None:
        self.registry = registry or _default_registry(web_loader=web_loader)
        self.max_concurrent = max(1, int(max_concurrent))

    def _resolve(self, source: str) -> Loader | None:
        return self.registry.resolve(source)

    async def ingest(self, sources: Iterable[str | Path]) -> list[Document]:
        """Run the pipeline. Returns documents sorted by source."""
        resolved_sources = _expand_sources(sources)
        if not resolved_sources:
            logger.warning("ingestion: no sources to process")
            return []

        sem = asyncio.Semaphore(self.max_concurrent)
        logger.info(
            "ingestion: processing %d source(s) with concurrency=%d",
            len(resolved_sources),
            self.max_concurrent,
        )

        async def _run_one(source: str) -> list[Document]:
            loader = self._resolve(source)
            if loader is None:
                logger.warning("no loader matched source: %s", source)
                return []
            async with sem:
                try:
                    docs = await loader.load(source)
                except Exception as exc:
                    logger.warning(
                        "loader %s failed on %s: %s", loader.name, source, exc
                    )
                    return []
            logger.info(
                "ingested: %s (%s, %d doc(s))", source, loader.name, len(docs)
            )
            return docs

        all_lists = await asyncio.gather(
            *[_run_one(s) for s in resolved_sources],
            return_exceptions=False,
        )
        docs: list[Document] = [d for sublist in all_lists for d in sublist]
        docs.sort(key=lambda d: d.source)
        logger.info("ingestion: produced %d document(s)", len(docs))
        return docs
