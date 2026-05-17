"""Tests for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from gyroscope.core.models import Document, DocumentKind
from gyroscope.ingestion.base import Loader, LoaderRegistry
from gyroscope.ingestion.pipeline import IngestionPipeline
from gyroscope.ingestion.web import WebLoader


class _ExplodingLoader(Loader):
    """Loader that always raises so we can prove failures are isolated."""

    name = "explode"

    def can_load(self, source: str) -> bool:
        return source.endswith(".explode")

    async def load(self, source: str) -> list[Document]:
        raise RuntimeError(f"intentional failure for {source}")


@pytest.mark.asyncio
async def test_pipeline_mixed_sources_ordering_and_failure_isolation(tmp_path: Path) -> None:
    # Build a small mixed corpus on disk.
    (tmp_path / "a.txt").write_text("alpha file\nbody.\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Bravo\n\nBody.\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("charlie\n", encoding="utf-8")
    bad = tmp_path / "evil.explode"
    bad.write_text("noop", encoding="utf-8")

    # Stub the web loader so we never touch the network.
    web = WebLoader(max_depth=0, max_pages=1, respect_robots=False)
    HTML = "<html><head><title>Stub</title></head><body><h1>X</h1><p>Body content abundant.</p></body></html>"

    async def stub_fetch(client: httpx.AsyncClient, url: str) -> str:
        return HTML

    web.fetch = stub_fetch  # type: ignore[method-assign]

    # Custom registry: include our exploding loader BEFORE the defaults so it wins.
    from gyroscope.ingestion.docx import DocxLoader
    from gyroscope.ingestion.html import HtmlLoader
    from gyroscope.ingestion.markdown import MarkdownLoader
    from gyroscope.ingestion.pdf import PdfLoader
    from gyroscope.ingestion.txt import TxtLoader

    registry = LoaderRegistry()
    registry.register(_ExplodingLoader())
    registry.register(PdfLoader())
    registry.register(MarkdownLoader())
    registry.register(HtmlLoader())
    registry.register(DocxLoader())
    registry.register(TxtLoader())
    registry.register(web)

    pipeline = IngestionPipeline(registry=registry, max_concurrent=4)

    sources: list[str | Path] = [
        tmp_path,  # directory expansion
        "https://example.com/page",  # web (stubbed)
        bad,  # exploding loader — must not kill batch
    ]
    docs = await pipeline.ingest(sources)

    # The exploding source must be skipped silently.
    assert all(not d.source.endswith(".explode") for d in docs)

    # We expect: a.txt, b.md, sub/c.txt, https://example.com/page
    kinds = sorted({d.kind for d in docs}, key=lambda k: k.value)
    assert DocumentKind.TXT in kinds
    assert DocumentKind.MARKDOWN in kinds
    assert DocumentKind.WEB in kinds
    assert len(docs) == 4

    # Output is deterministically sorted by source string.
    assert [d.source for d in docs] == sorted(d.source for d in docs)


@pytest.mark.asyncio
async def test_pipeline_returns_empty_when_no_sources(tmp_path: Path) -> None:
    pipeline = IngestionPipeline()
    docs = await pipeline.ingest([])
    assert docs == []


@pytest.mark.asyncio
async def test_pipeline_skips_unknown_source(tmp_path: Path) -> None:
    pipeline = IngestionPipeline()
    # A file with an unrecognised extension should be skipped without crashing.
    weird = tmp_path / "weird.bin"
    weird.write_bytes(b"\x00\x01\x02")
    docs = await pipeline.ingest([weird])
    assert docs == []


@pytest.mark.asyncio
async def test_pipeline_glob_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    pipeline = IngestionPipeline()
    docs = await pipeline.ingest(["*.txt"])
    sources = sorted(Path(d.source).name for d in docs)
    assert sources == ["one.txt", "two.txt"]
