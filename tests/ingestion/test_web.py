"""Tests for the web loader using respx + a fetch monkeypatch.

We never hit the network: ``WebLoader.fetch`` is monkeypatched to return
canned HTML, and robots checking is disabled for the test loader so the
crawl does not try to fetch ``robots.txt``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from gyroscope.core.models import DocumentKind
from gyroscope.ingestion.web import WebLoader

HTML_HOME = """
<html><head><title>Home Page</title>
<meta name="description" content="example">
</head><body>
<h1>Welcome</h1>
<p>Hello there, this is the homepage with enough text to be extracted by trafilatura cleanly.</p>
<a href="/about">About us</a>
<a href="https://other.example.org/x">External</a>
</body></html>
"""

HTML_ABOUT = """
<html><head><title>About Us</title></head><body>
<h1>About</h1>
<p>This page explains the project's mission and includes substantial body text so the extractor recognises it as real content.</p>
</body></html>
"""


def _install_fetch_stub(loader: WebLoader, pages: dict[str, str]) -> None:
    async def stub_fetch(client: httpx.AsyncClient, url: str) -> str:
        if url not in pages:
            raise httpx.HTTPError(f"not found: {url}")
        return pages[url]

    loader.fetch = stub_fetch  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_web_loader_single_url() -> None:
    loader = WebLoader(max_depth=0, max_pages=1, respect_robots=False)
    _install_fetch_stub(loader, {"https://example.com/": HTML_HOME})

    docs = await loader.load("https://example.com/")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.kind == DocumentKind.WEB
    assert doc.source == "https://example.com/"
    assert doc.metadata["url"] == "https://example.com/"
    assert "scrape_date" in doc.metadata
    assert "Hello there" in doc.text
    # Title comes from trafilatura metadata, falling back to <title>.
    assert doc.title in {"Home Page", "Welcome"}


@pytest.mark.asyncio
async def test_web_loader_recursive_same_host() -> None:
    loader = WebLoader(max_depth=1, max_pages=5, respect_robots=False, concurrency=2)
    _install_fetch_stub(
        loader,
        {
            "https://example.com/": HTML_HOME,
            "https://example.com/about": HTML_ABOUT,
        },
    )
    docs = await loader.load_many(["https://example.com/"])
    sources = [d.source for d in docs]
    assert "https://example.com/" in sources
    assert "https://example.com/about" in sources
    # External host must be filtered out.
    assert not any("other.example.org" in s for s in sources)
    # Stable ordering by source.
    assert sources == sorted(sources)


@pytest.mark.asyncio
async def test_web_loader_max_pages_caps_crawl() -> None:
    loader = WebLoader(max_depth=5, max_pages=1, respect_robots=False)
    _install_fetch_stub(
        loader,
        {
            "https://example.com/": HTML_HOME,
            "https://example.com/about": HTML_ABOUT,
        },
    )
    docs = await loader.load_many(["https://example.com/"])
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_web_loader_handles_fetch_failure_gracefully() -> None:
    loader = WebLoader(max_depth=0, max_pages=1, respect_robots=False)

    async def stub_fetch(client: httpx.AsyncClient, url: str) -> str:
        raise httpx.ConnectError("boom")

    loader.fetch = stub_fetch  # type: ignore[method-assign]
    docs = await loader.load("https://example.com/")
    assert docs == []


@respx.mock
@pytest.mark.asyncio
async def test_web_loader_with_respx_full_stack() -> None:
    """Exercise the real ``fetch`` path with respx as a transport."""
    respx.get("https://example.com/robots.txt").mock(
        return_value=httpx.Response(404, text="not found")
    )
    respx.get("https://example.com/").mock(
        return_value=httpx.Response(200, html=HTML_HOME)
    )

    loader = WebLoader(max_depth=0, max_pages=1, respect_robots=True)
    docs = await loader.load("https://example.com/")
    assert len(docs) == 1
    assert "Hello there" in docs[0].text


@respx.mock
@pytest.mark.asyncio
async def test_web_loader_respects_robots_disallow() -> None:
    robots = "User-agent: *\nDisallow: /private/\n"
    respx.get("https://example.com/robots.txt").mock(
        return_value=httpx.Response(200, text=robots)
    )

    loader = WebLoader(max_depth=0, max_pages=1, respect_robots=True)
    # Should not attempt the disallowed URL — fetch would otherwise need a mock.
    docs = await loader.load("https://example.com/private/secret")
    assert docs == []


@pytest.mark.asyncio
async def test_web_loader_can_load_only_urls() -> None:
    loader = WebLoader()
    assert loader.can_load("http://example.com")
    assert loader.can_load("https://example.com/path")
    assert not loader.can_load("/tmp/foo.html")
    assert not loader.can_load("foo.pdf")


# Silence unused warning for ``Any`` import in some linters.
_: Any = None
