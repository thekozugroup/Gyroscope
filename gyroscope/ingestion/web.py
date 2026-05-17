"""Web loader.

Fetches one or more URLs with ``httpx``, extracts the main content with
``trafilatura`` and respects ``robots.txt``. Optionally crawls recursively
within the same host up to ``max_depth`` and ``max_pages``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura
from bs4 import BeautifulSoup

from gyroscope.core.logging import get_logger
from gyroscope.core.models import Document, DocumentKind
from gyroscope.ingestion.base import Loader, is_url

logger = get_logger(__name__)


USER_AGENT = "GyroscopeBot/0.1 (+https://github.com/kozugroup/gyroscope)"
DEFAULT_TIMEOUT = 30.0
DEFAULT_CONCURRENCY = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_url(url: str) -> str:
    """Drop the fragment so duplicates do not get crawled twice."""
    return urldefrag(url)[0]


def _same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "javascript:", "tel:", "#")):
            continue
        absolute = _normalise_url(urljoin(base_url, href))
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        out.append(absolute)
    return out


def _extract_content(html: str, url: str) -> tuple[str, str | None, str | None]:
    """Run trafilatura and return ``(text, title, sitename)``."""
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    ) or ""
    title: str | None = None
    sitename: str | None = None
    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
    except Exception:
        meta = None
    if meta is not None:
        title = getattr(meta, "title", None) or None
        sitename = getattr(meta, "sitename", None) or None
    if not title:
        soup = BeautifulSoup(html, "html.parser")
        tt = soup.find("title")
        if tt:
            title = tt.get_text(strip=True) or None
    return text.strip() + ("\n" if text else ""), title, sitename


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class WebLoader(Loader):
    """Loader for ``http://`` / ``https://`` URLs with optional recursive crawl."""

    name = "web"

    def __init__(
        self,
        *,
        max_depth: int = 0,
        max_pages: int = 1,
        concurrency: int = DEFAULT_CONCURRENCY,
        timeout: float = DEFAULT_TIMEOUT,
        same_host_only: bool = True,
        respect_robots: bool = True,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.max_depth = max(0, int(max_depth))
        self.max_pages = max(1, int(max_pages))
        self.concurrency = max(1, int(concurrency))
        self.timeout = float(timeout)
        self.same_host_only = bool(same_host_only)
        self.respect_robots = bool(respect_robots)
        self.user_agent = user_agent
        # robots.txt cache: host -> RobotFileParser (or None if fetch failed)
        self._robots_cache: dict[str, RobotFileParser | None] = {}

    # ----- registry contract -----

    def can_load(self, source: str) -> bool:
        return is_url(source)

    # ----- public API -----

    async def load(self, source: str) -> list[Document]:
        """Fetch ``source`` (single URL) and, when configured, crawl its links."""
        return await self.load_many([source])

    async def load_many(self, sources: list[str]) -> list[Document]:
        """Fetch a batch of seed URLs and return discovered :class:`Document`."""
        seeds = [_normalise_url(s) for s in sources if is_url(s)]
        if not seeds:
            return []

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        ) as client:
            return await self._crawl(client, seeds)

    # ----- network primitives (overridable for tests) -----

    async def fetch(self, client: httpx.AsyncClient, url: str) -> str:
        """Fetch ``url`` and return the response body as text."""
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text

    async def _check_robots(self, client: httpx.AsyncClient, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots_cache.get(host)
        if rp is None and host not in self._robots_cache:
            rp = await self._load_robots(client, host)
            self._robots_cache[host] = rp
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    async def _load_robots(self, client: httpx.AsyncClient, host: str) -> RobotFileParser | None:
        robots_url = f"{host}/robots.txt"
        try:
            resp = await client.get(robots_url)
        except Exception as exc:
            logger.debug("robots.txt fetch failed for %s: %s", host, exc)
            return None
        if resp.status_code >= 400:
            return None
        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp

    # ----- crawl -----

    async def _crawl(self, client: httpx.AsyncClient, seeds: list[str]) -> list[Document]:
        visited: set[str] = set()
        queue: deque[tuple[str, int, str]] = deque(
            (url, 0, url) for url in seeds
        )  # (url, depth, seed_host_url)
        sem = asyncio.Semaphore(self.concurrency)
        results: list[Document] = []

        async def _process(url: str, depth: int, seed: str) -> list[str]:
            if not await self._check_robots(client, url):
                logger.info("skipping (robots): %s", url)
                return []
            async with sem:
                try:
                    html = await self.fetch(client, url)
                except Exception as exc:
                    logger.warning("fetch failed: %s (%s)", url, exc)
                    return []
            text, title, sitename = _extract_content(html, url)
            metadata: dict[str, Any] = {
                "url": url,
                "title": title,
                "scrape_date": datetime.now(UTC).isoformat(),
                "sitename": sitename,
                "depth": depth,
            }
            doc = Document(
                source=url,
                kind=DocumentKind.WEB,
                text=text,
                title=title,
                metadata=metadata,
            )
            results.append(doc)
            if depth >= self.max_depth:
                return []
            children: list[str] = []
            for link in _extract_links(html, url):
                if self.same_host_only and not _same_host(link, seed):
                    continue
                children.append(link)
            return children

        while queue and len(visited) < self.max_pages:
            batch: list[tuple[str, int, str]] = []
            while queue and len(batch) < self.concurrency and (len(visited) + len(batch)) < self.max_pages:
                url, depth, seed = queue.popleft()
                url = _normalise_url(url)
                if url in visited:
                    continue
                visited.add(url)
                batch.append((url, depth, seed))

            if not batch:
                break

            children_lists = await asyncio.gather(
                *[_process(u, d, s) for u, d, s in batch],
                return_exceptions=False,
            )
            for (_, depth, seed), children in zip(batch, children_lists, strict=True):
                for link in children:
                    if link in visited:
                        continue
                    queue.append((link, depth + 1, seed))

        # Stable ordering by URL for determinism.
        results.sort(key=lambda d: d.source)
        return results
