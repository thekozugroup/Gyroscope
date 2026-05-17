"""Loader interface and registry.

Loaders translate a *source string* (a filesystem path or a URL) into one or
more :class:`Document` instances. Each concrete loader advertises what it can
handle via :meth:`Loader.can_load`; the :class:`LoaderRegistry` picks the
first registered loader that says yes.

The registry preserves insertion order, so more specific loaders should be
registered before more general ones (the pipeline does this automatically).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import metadata as importlib_metadata
from typing import ClassVar
from urllib.parse import urlparse

from gyroscope.core.logging import get_logger
from gyroscope.core.models import Document

logger = get_logger(__name__)


_DEFAULT_LOADER_ENTRY_POINT_GROUP = "gyroscope.loaders"


class Loader(ABC):
    """Abstract base class for ingestion loaders."""

    name: ClassVar[str] = "loader"

    @abstractmethod
    def can_load(self, source: str) -> bool:
        """Return True if this loader is able to handle ``source``."""

    @abstractmethod
    async def load(self, source: str) -> list[Document]:
        """Resolve ``source`` to one or more :class:`Document` objects."""


def is_url(source: str) -> bool:
    """Return True if ``source`` looks like an http(s) URL."""
    try:
        parsed = urlparse(source)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class LoaderRegistry:
    """Ordered registry that resolves a source string to the right Loader."""

    def __init__(self) -> None:
        self._loaders: list[Loader] = []

    def register(self, loader: Loader) -> None:
        """Append a loader to the registry."""
        self._loaders.append(loader)

    def loaders(self) -> list[Loader]:
        return list(self._loaders)

    def resolve(self, source: str) -> Loader | None:
        """Return the first loader that claims it can handle ``source``."""
        for loader in self._loaders:
            try:
                if loader.can_load(source):
                    return loader
            except Exception:
                logger.debug("loader %s raised during can_load(%r)", loader.name, source)
                continue
        return None

    @classmethod
    def discover(cls, group: str = _DEFAULT_LOADER_ENTRY_POINT_GROUP) -> LoaderRegistry:
        """Build a registry from installed ``importlib.metadata`` entry points.

        Each entry point in ``group`` (default ``"gyroscope.loaders"``) must
        resolve to a zero-argument callable that returns a :class:`Loader`
        instance. Failures during discovery — missing entry points, importable
        targets that raise on load, or callables that return non-``Loader``
        values — are logged at DEBUG and skipped so a broken third-party
        plugin can never poison the host pipeline.
        """
        reg = cls()
        try:
            entry_points = importlib_metadata.entry_points(group=group)
        except TypeError:
            # Older importlib.metadata returned a dict-like object without
            # the ``group=`` keyword. We don't ship on that runtime, but
            # be defensive so plugin discovery never crashes the pipeline.
            entry_points = importlib_metadata.entry_points().get(group, [])  # type: ignore[arg-type]
        for ep in entry_points:
            try:
                factory = ep.load()
                loader = factory()
            except Exception:
                logger.debug("loader entry point %r failed to load", getattr(ep, "name", ep))
                continue
            if not isinstance(loader, Loader):
                logger.debug(
                    "loader entry point %r returned non-Loader %r",
                    getattr(ep, "name", ep),
                    type(loader).__name__,
                )
                continue
            reg.register(loader)
        return reg
