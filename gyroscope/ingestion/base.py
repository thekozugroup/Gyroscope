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
from typing import ClassVar
from urllib.parse import urlparse

from gyroscope.core.logging import get_logger
from gyroscope.core.models import Document

logger = get_logger(__name__)


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
