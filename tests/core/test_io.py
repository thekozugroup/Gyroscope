"""Tests for IO helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from gyroscope.core.io import golden_to_markdown, read_jsonl, read_yaml, write_jsonl, write_yaml
from gyroscope.core.models import GoldenDocument, Identity


def test_jsonl_roundtrip(tmp_path: Path):
    rows = [{"a": 1}, {"a": 2, "b": "x"}]
    n = write_jsonl(tmp_path / "out.jsonl", rows)
    assert n == 2
    back = list(read_jsonl(tmp_path / "out.jsonl"))
    assert back == rows


def test_jsonl_creates_parent(tmp_path: Path):
    target = tmp_path / "nested/dir/out.jsonl"
    write_jsonl(target, [{"x": 1}])
    assert target.exists()


def test_yaml_roundtrip(tmp_path: Path):
    obj = {"a": [1, 2, 3], "nested": {"k": "v"}}
    write_yaml(tmp_path / "out.yaml", obj)
    assert read_yaml(tmp_path / "out.yaml") == obj


def test_golden_to_markdown_minimal():
    g = GoldenDocument(
        identity=Identity(role="X", description="d", mission="m"),
    )
    md = golden_to_markdown(g)
    assert "# Identity" in md
    assert "# Mission" in md
    # empty sections are omitted
    assert "# Principles" not in md
    assert "# Procedures" not in md


def test_golden_to_markdown_type_check():
    with pytest.raises(TypeError):
        golden_to_markdown({"not": "a golden"})
