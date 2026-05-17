"""Tests for ``LoaderRegistry.discover()`` and the default-registry merge.

We never touch the real installed environment — every test patches
``importlib.metadata.entry_points`` with a hand-crafted entry point so the
registry behaviour is fully deterministic.
"""

from __future__ import annotations

from typing import Any

import pytest

from gyroscope.core.models import Document, DocumentKind
from gyroscope.ingestion import base as base_mod
from gyroscope.ingestion import pipeline as pipeline_mod
from gyroscope.ingestion.base import Loader, LoaderRegistry


class _SentinelLoader(Loader):
    """A trivial loader used to detect that discovery wired it in."""

    name = "sentinel-loader"

    def can_load(self, source: str) -> bool:
        return source.startswith("sentinel://")

    async def load(self, source: str) -> list[Document]:
        return [
            Document(source=source, kind=DocumentKind.TXT, text="sentinel-payload")
        ]


class _FakeEntryPoint:
    """Minimal shim with the bits ``LoaderRegistry.discover`` reads."""

    def __init__(self, name: str, factory: Any) -> None:
        self.name = name
        self._factory = factory

    def load(self) -> Any:
        return self._factory


def test_discover_with_no_entry_points_returns_empty_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean environment with no registered entry points must not fail."""

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return []

    monkeypatch.setattr(base_mod.importlib_metadata, "entry_points", fake_entry_points)
    reg = LoaderRegistry.discover()
    assert isinstance(reg, LoaderRegistry)
    assert reg.loaders() == []


def test_discover_registers_loaders_from_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A monkey-patched entry point whose factory returns a ``Loader`` must
    be registered in the discovered registry."""

    ep = _FakeEntryPoint(name="sentinel", factory=lambda: _SentinelLoader())

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        assert group == "gyroscope.loaders"
        return [ep]

    monkeypatch.setattr(base_mod.importlib_metadata, "entry_points", fake_entry_points)
    reg = LoaderRegistry.discover()
    loaders = reg.loaders()
    assert len(loaders) == 1
    assert isinstance(loaders[0], _SentinelLoader)
    # And the resolver routes the matching scheme through it.
    resolved = reg.resolve("sentinel://thing")
    assert isinstance(resolved, _SentinelLoader)


def test_discover_skips_broken_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry point whose factory raises must not poison the registry —
    the well-behaved entry point still registers."""

    def boom_factory() -> Loader:
        raise RuntimeError("bad plugin")

    eps = [
        _FakeEntryPoint(name="boom", factory=boom_factory),
        _FakeEntryPoint(name="good", factory=lambda: _SentinelLoader()),
    ]

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return eps

    monkeypatch.setattr(base_mod.importlib_metadata, "entry_points", fake_entry_points)
    reg = LoaderRegistry.discover()
    loaders = reg.loaders()
    assert len(loaders) == 1
    assert isinstance(loaders[0], _SentinelLoader)


def test_discover_skips_non_loader_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry points that return something other than a ``Loader`` are
    silently skipped — they would be a contract violation that we should
    never honour at runtime."""

    eps = [
        _FakeEntryPoint(name="not-a-loader", factory=lambda: object()),
    ]

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return eps

    monkeypatch.setattr(base_mod.importlib_metadata, "entry_points", fake_entry_points)
    reg = LoaderRegistry.discover()
    assert reg.loaders() == []


def test_default_registry_merges_discovered_after_built_ins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_default_registry`` must register the built-ins first (so canonical
    file extensions never get hijacked by a plugin) and then append the
    entry-point discoveries."""
    ep = _FakeEntryPoint(name="sentinel", factory=lambda: _SentinelLoader())

    def fake_entry_points(*, group: str) -> list[_FakeEntryPoint]:
        return [ep]

    monkeypatch.setattr(base_mod.importlib_metadata, "entry_points", fake_entry_points)

    reg = pipeline_mod._default_registry()
    loaders = reg.loaders()
    # Built-ins come first.
    builtin_names = {type(loader).__name__ for loader in loaders[:-1]}
    assert {
        "PdfLoader",
        "MarkdownLoader",
        "HtmlLoader",
        "DocxLoader",
        "TxtLoader",
        "WebLoader",
    }.issubset(builtin_names)
    # Discovered loader is appended at the tail.
    assert isinstance(loaders[-1], _SentinelLoader)


def test_default_registry_survives_discovery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``LoaderRegistry.discover`` itself raises, ingestion must still
    succeed with only the built-in loaders — a broken plugin is never
    allowed to block the host pipeline."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("registry discovery exploded")

    monkeypatch.setattr(LoaderRegistry, "discover", classmethod(boom))
    reg = pipeline_mod._default_registry()
    names = {type(loader).__name__ for loader in reg.loaders()}
    assert {
        "PdfLoader",
        "MarkdownLoader",
        "HtmlLoader",
        "DocxLoader",
        "TxtLoader",
        "WebLoader",
    }.issubset(names)
