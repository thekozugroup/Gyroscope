"""Format registry: custom formats register and dispatch through render()."""

from __future__ import annotations

from typing import Any

import pytest

from gyroscope.core.models import Trajectory, TrajectoryMessage
from gyroscope.sft.formats import (
    FORMAT_READERS,
    FORMAT_WRITERS,
    register_format,
    render,
    to_alpaca,
    to_chatml,
    to_sharegpt,
)

from .conftest import make_trajectory


@pytest.fixture
def restore_registry():
    """Snapshot the registry and restore it after the test so registrations
    do not leak between tests."""
    writers_snapshot = dict(FORMAT_WRITERS)
    readers_snapshot = dict(FORMAT_READERS)
    yield
    FORMAT_WRITERS.clear()
    FORMAT_WRITERS.update(writers_snapshot)
    FORMAT_READERS.clear()
    FORMAT_READERS.update(readers_snapshot)


def test_builtin_formats_are_registered():
    assert FORMAT_WRITERS["sharegpt"] is to_sharegpt
    assert FORMAT_WRITERS["chatml"] is to_chatml
    assert FORMAT_WRITERS["alpaca"] is to_alpaca
    # alpaca is lossy and therefore intentionally has no reader.
    assert "alpaca" not in FORMAT_READERS
    assert set(FORMAT_READERS) == {"sharegpt", "chatml"}


def test_render_dispatches_to_registered_custom_format(restore_registry):
    def to_minimal(traj: Trajectory) -> dict[str, Any]:
        return {"tag": "minimal", "id": traj.id}

    register_format("minimal", to_minimal)
    assert "minimal" in FORMAT_WRITERS

    traj = make_trajectory()
    rendered = render(traj, "minimal")
    assert rendered == {"tag": "minimal", "id": traj.id}


def test_register_format_accepts_optional_reader(restore_registry):
    def to_minimal(traj: Trajectory) -> dict[str, Any]:
        return {"id": traj.id, "system": traj.system}

    def from_minimal(row: dict[str, Any]) -> Trajectory:
        # Just enough to round-trip the id; full reverse parsing isn't the
        # point of this test.
        return Trajectory(
            id=row["id"],
            scenario_id="SCN-roundtrip",
            system=row.get("system", ""),
            messages=[],
            tags={},
        )

    register_format("minimal-rt", to_minimal, reader=from_minimal)
    assert "minimal-rt" in FORMAT_WRITERS
    assert FORMAT_READERS["minimal-rt"] is from_minimal


def test_register_format_without_reader_does_not_pollute_readers(restore_registry):
    register_format("writer-only", lambda traj: {"id": traj.id})
    assert "writer-only" in FORMAT_WRITERS
    assert "writer-only" not in FORMAT_READERS


def test_render_unknown_format_raises_with_registered_list():
    with pytest.raises(ValueError) as excinfo:
        render(make_trajectory(), "definitely-not-a-format")
    msg = str(excinfo.value)
    assert "definitely-not-a-format" in msg
    # The error must surface the registered formats so users can self-correct.
    assert "sharegpt" in msg
    assert "chatml" in msg


def test_register_format_overrides_existing_binding(restore_registry):
    """Overwriting an existing name lets tests swap implementations safely."""
    def replacement(traj: Trajectory) -> dict[str, Any]:
        return {"replaced": True, "id": traj.id}

    register_format("sharegpt", replacement)
    rendered = render(make_trajectory(), "sharegpt")
    assert rendered == {"replaced": True, "id": rendered["id"]}


def test_eval_pipeline_uses_registry(restore_registry, tmp_path):
    """End-to-end: register a format and EvalPipeline accepts it."""
    from gyroscope.eval.pipeline import EvalPipeline

    def to_minimal(traj: Trajectory) -> dict[str, Any]:
        return {"id": traj.id, "tag": "minimal"}

    register_format("minimal-eval", to_minimal)

    pipe = EvalPipeline("minimal-eval")  # constructor now reads registry
    ev = [
        Trajectory(
            id="TRJ-0001",
            scenario_id="SCN-0001",
            system="s",
            messages=[
                TrajectoryMessage(role="user", content="u"),
                TrajectoryMessage(role="assistant", content="a"),
            ],
            tags={"procedure_ids": ["PRC-0001"]},
        ),
    ]
    out_path = pipe.write(ev, train_trajectories=[], output_dir=tmp_path)
    assert out_path.exists()
    written = out_path.read_text(encoding="utf-8").strip()
    assert '"tag": "minimal"' in written
