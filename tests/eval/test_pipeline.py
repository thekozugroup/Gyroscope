"""Tests for the eval-set writer + leakage check."""

from __future__ import annotations

from pathlib import Path

import pytest

from gyroscope.core.models import Trajectory, TrajectoryMessage
from gyroscope.eval.pipeline import EvalPipeline, leakage_check


def _t(idx: int, procedure_id: str) -> Trajectory:
    return Trajectory(
        id=f"TRJ-{idx:04d}",
        scenario_id=f"SCN-{idx:04d}",
        system="sys",
        messages=[
            TrajectoryMessage(role="user", content=f"u{idx}"),
            TrajectoryMessage(role="assistant", content=f"a{idx}"),
        ],
        tags={"procedure_ids": [procedure_id], "persona": "p", "difficulty": "easy"},
    )


def test_leakage_check_empty_when_disjoint():
    train = [_t(1, "PRC-0001"), _t(2, "PRC-0002")]
    ev = [_t(3, "PRC-0003")]
    assert leakage_check(train, ev) == set()


def test_leakage_check_detects_overlap():
    train = [_t(1, "PRC-0001")]
    ev = [_t(2, "PRC-0001")]
    assert leakage_check(train, ev) == {"PRC-0001"}


def test_write_raises_on_leakage_strict(tmp_path: Path):
    pipe = EvalPipeline("sharegpt")
    train = [_t(1, "PRC-0001")]
    ev = [_t(2, "PRC-0001")]
    with pytest.raises(ValueError):
        pipe.write(ev, train, tmp_path, strict=True)


def test_write_warns_on_leakage_non_strict(tmp_path: Path, caplog):
    pipe = EvalPipeline("sharegpt")
    train = [_t(1, "PRC-0001")]
    ev = [_t(2, "PRC-0001")]
    out = pipe.write(ev, train, tmp_path, strict=False)
    assert out.exists()
    assert any("leakage" in r.message.lower() for r in caplog.records)
