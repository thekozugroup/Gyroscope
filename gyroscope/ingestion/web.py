"""Web loader.

Fetches one or more URLs with ``httpx``, extracts the main content with
``trafilatura`` and respects ``robots.txt``. Optionally crawls recursively
within the same host up to ``max_depth`` and ``max_pages``.

Safety properties:

- Redirects are NOT followed automatically. The loader inspects each 3xx
  ``Location`` header, blocks redirects whose target host resolves to a
  loopback / link-local / RFC1918 address (or names like ``localhost`` /
  ``*.local``), and — when ``same_host_only`` is set — refuses redirects
  that leave the seed host.
- The ``robots.txt`` cache is keyed on host with in-flight
  :class:`asyncio.Future` placeholders, so concurrent fetches for the same
  host coalesce into a single network round-trip.
"""

from __future__ import annotations

import asyncio
import ipaddress
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
MAX_REDIRECTS = 5


_BLOCKED_HOSTNAMES: frozenset[str] = frozenset({"localhost", "ip6-localhost", "ip6-loopback"})


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
        href = str(a.get("href") or "").strip()
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
    text = (
        trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        or ""
    )
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


def _is_blocked_host(host: str) -> bool:
    """Return True iff ``host`` resolves to a private / loopback / link-local target.

    Only string-level inspection is performed: IPs are parsed with
    :mod:`ipaddress` and hostnames are checked against a small denylist
    (``localhost``, ``*.local``). DNS lookups are intentionally NOT done — the
    caller is expected to disable redirects to suspicious hostnames before
    issuing any network request.
    """
    if not host:
        return True
    host = host.lower().strip()
    # Strip optional ``[ipv6]`` brackets and port.
    if host.startswith("[") and "]" in host:
        host = host[1 : host.index("]")]
    elif ":" in host and host.count(":") == 1:
        # IPv4:port — drop port.
        host = host.split(":", 1)[0]
    if host in _BLOCKED_HOSTNAMES:
        return True
    if host.endswith(".local") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


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
        # robots.txt cache: host -> Future[RobotFileParser | None]. Storing
        # futures (not parsers) lets concurrent callers for the same host
        # coalesce on a single in-flight fetch.
        self._robots_cache: dict[str, asyncio.Future[RobotFileParser | None]] = {}
        self._robots_lock: asyncio.Lock | None = None

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
            follow_redirects=False,
            headers={"User-Agent": self.user_agent},
        ) as client:
            return await self._crawl(client, seeds)

    # ----- network primitives (overridable for tests) -----

    async def fetch(self, client: httpx.AsyncClient, url: str) -> str:
        """Fetch ``url`` (handling a bounded redirect chain) and return body text.

        Redirects are inspected manually: off-host (when ``same_host_only`` is
        set) or private-network targets cause the chain to stop and a warning
        to be logged. The final non-3xx response is returned.
        """
        current = url
        origin_host = urlparse(url).netloc
        for _ in range(MAX_REDIRECTS + 1):
            resp = await client.get(current)
            if resp.status_code in {301, 302, 303, 307, 308}:
                location = resp.headers.get("location")
                if not location:
                    resp.raise_for_status()
                    return resp.text
                target = _normalise_url(urljoin(current, location))
                target_host = urlparse(target).netloc
                if _is_blocked_host(target_host):
                    logger.warning(
                        "refusing redirect to blocked host: %s -> %s",
                        current,
                        target,
                    )
                    raise httpx.HTTPError(f"redirect to blocked host refused: {target}")
                if self.same_host_only and target_host.lower() != origin_host.lower():
                    logger.warning(
                        "refusing off-host redirect (%s -> %s)",
                        current,
                        target,
                    )
                    raise httpx.HTTPError(f"off-host redirect refused: {target}")
                current = target
                continue
            resp.raise_for_status()
            return resp.text
        raise httpx.HTTPError(f"too many redirects starting at {url}")

    async def _check_robots(self, client: httpx.AsyncClient, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        future = await self._get_or_create_robots_future(client, host)
        rp = await future
        if rp is None:
            return True
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    async def _get_or_create_robots_future(
        self, client: httpx.AsyncClient, host: str
    ) -> asyncio.Future[RobotFileParser | None]:
        """Return a Future resolving to the parsed robots.txt for ``host``.

        Subsequent calls for the same host (even while the first fetch is in
        flight) reuse the existing Future, so robots.txt is fetched exactly
        once per host per loader lifetime.
        """
        if self._robots_lock is None:
            self._robots_lock = asyncio.Lock()
        async with self._robots_lock:
            existing = self._robots_cache.get(host)
            if existing is not None:
                return existing
            loop = asyncio.get_running_loop()
            future: asyncio.Future[RobotFileParser | None] = loop.create_future()
            self._robots_cache[host] = future
        # Fetch outside the lock so concurrent hosts don't serialise.
        try:
            parser = await self._load_robots(client, host)
        except Exception as exc:
            # Failure must not poison the cache; treat as "no robots.txt".
            logger.debug("robots.txt fetch crashed for %s: %s", host, exc)
            parser = None
        if not future.done():
            future.set_result(parser)
        return future

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

        # Sliding-window worker pool. The old "build a batch then gather"
        # loop stalled the whole batch on the slowest fetch — under variable
        # latency, effective concurrency dipped well below ``self.concurrency``.
        # Now we keep up to ``self.concurrency`` fetches in flight at all times,
        # replenishing each completed slot from the queue immediately.
        in_flight: dict[asyncio.Task[list[str]], tuple[str, int, str]] = {}

        def _enqueue_next() -> bool:
            """Pop the next unvisited URL and schedule it; return True on success."""
            while queue and len(visited) < self.max_pages:
                url, depth, seed = queue.popleft()
                url = _normalise_url(url)
                if url in visited:
                    continue
                visited.add(url)
                task = asyncio.create_task(_process(url, depth, seed))
                in_flight[task] = (url, depth, seed)
                return True
            return False

        # Prime the window with up to ``self.concurrency`` initial fetches.
        for _ in range(self.concurrency):
            if not _enqueue_next():
                break

        while in_flight:
            done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                _, depth, _seed = in_flight.pop(task)
                try:
                    children = task.result()
                except Exception as exc:
                    logger.warning("WebLoader fetch task raised: %s", exc)
                    children = []
                for link in children:
                    if link in visited:
                        continue
                    queue.append((link, depth + 1, _seed))
                # Refill the freed slot.
                _enqueue_next()

        # Stable ordering by URL for determinism.
        results.sort(key=lambda d: d.source)
        return results
