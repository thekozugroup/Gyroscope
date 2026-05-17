"""CLI smoke tests.

These tests do not run any real phase — they verify the Typer ``app`` is
importable, every subcommand exposes ``--help`` without crashing (which
catches import-time errors in any phase module the CLI depends on), and the
``report`` subcommand can round-trip an SFT dataset written via
``to_sharegpt`` back through ``from_sharegpt`` and emit the three report
artefacts on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from gyroscope.cli import app
from gyroscope.core.models import (
    Document,
    DocumentKind,
    GoldenDocument,
    Identity,
    Principle,
    Trajectory,
    TrajectoryMessage,
)
from gyroscope.sft.formats import to_sharegpt

# Every subcommand the CLI exposes. Keep this aligned with @app.command()s in
# gyroscope/cli.py — if a new subcommand is added without updating this list
# the smoke test fails, forcing the author to think about its --help shape.
SUBCOMMANDS = ("ingest", "curate", "sft", "rewards", "report", "run")


def test_app_help_lists_every_subcommand() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stdout
    for sub in SUBCOMMANDS:
        assert sub in result.stdout, (
            f"subcommand {sub!r} missing from --help output:\n{result.stdout}"
        )


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help_exits_zero(subcommand: str) -> None:
    runner = CliRunner()
    result = runner.invoke(app, [subcommand, "--help"])
    assert result.exit_code == 0, (
        f"`{subcommand} --help` exited {result.exit_code}: {result.stdout}"
    )


def _write_min_run(tmp_path: Path) -> Path:
    """Build a minimal but valid run directory the report subcommand can read."""
    run = tmp_path / "run"
    run.mkdir()

    # 1. documents.jsonl — one Document, just enough text to score against.
    doc = Document(
        source=str(run / "src.txt"),
        kind=DocumentKind.TXT,
        text=(
            "A principle about measurement: always verify before reporting. "
            "Quantities must be cross-checked."
        ),
    )
    (run / "documents.jsonl").write_text(
        json.dumps(doc.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    # 2. golden.json — minimal valid GoldenDocument.
    golden = GoldenDocument(
        identity=Identity(
            role="Measurement Expert",
            description="Verifies measurements before reporting.",
            mission="Report accurate quantities.",
        ),
        principles=[
            Principle(
                id="PRN-0001",
                statement="Always verify before reporting.",
                source_chunk_ids=[],
            ),
        ],
    )
    (run / "golden.json").write_text(golden.model_dump_json(), encoding="utf-8")

    # 3. sft.jsonl — one trajectory round-tripped through to_sharegpt.
    traj = Trajectory(
        id="TRJ-0001",
        scenario_id="SCN-0001",
        system="You verify measurements.",
        messages=[
            TrajectoryMessage(role="user", content="How do I report a quantity?"),
            TrajectoryMessage(
                role="assistant",
                content="Verify it against the source, then report.",
            ),
        ],
        tags={
            "procedure_ids": [],
            "principle_ids": ["PRN-0001"],
            "persona": "PER-0001",
            "difficulty": "medium",
        },
        quality_score=0.9,
    )
    (run / "sft.jsonl").write_text(json.dumps(to_sharegpt(traj)) + "\n", encoding="utf-8")

    # 4. rewards/reward_spec.yaml — empty specs list is a valid bundle for
    #    the report command (it just produces a 0 reward_soundness score).
    rewards_dir = run / "rewards"
    rewards_dir.mkdir()
    (rewards_dir / "reward_spec.yaml").write_text(yaml.safe_dump({"specs": []}), encoding="utf-8")

    return run


def test_report_subcommand_round_trips_through_from_sharegpt(tmp_path: Path) -> None:
    run = _write_min_run(tmp_path)

    runner = CliRunner()
    # threshold=0 means every axis >= 0 → all pass → exit 0.
    result = runner.invoke(
        app,
        ["report", "--run", str(run), "--threshold", "0"],
    )

    assert result.exit_code == 0, (
        f"report exited {result.exit_code}; stdout:\n{result.stdout}\nexc: {result.exception!r}"
    )
    # All three artefacts must land on disk.
    assert (run / "report.md").exists(), "report.md missing"
    assert (run / "report.html").exists(), "report.html missing"
    assert (run / "report.json").exists(), "report.json missing"

    # report.json must be valid JSON describing the axes the report covers.
    payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert "axes" in payload and isinstance(payload["axes"], dict)
    assert payload["axes"], "report.json should have at least one axis"


def test_run_subcommand_drives_autonomous_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`gyroscope run` constructs an AutonomousRunner, executes it, and
    surfaces the iteration table + exit code from the resulting report."""
    import gyroscope.runner as runner_mod

    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, cfg, *, threshold: float, max_iterations: int) -> None:
            captured["cfg_threshold"] = threshold
            captured["cfg_max_iter"] = max_iterations
            captured["input_paths"] = list(cfg.input_paths)
            captured["output_dir"] = cfg.output_dir
            captured["n_trajectories"] = cfg.sft.n_trajectories
            captured["reward_budget"] = cfg.rewards.reward_budget
            from gyroscope.quality.metrics import AxisScore, QualityReport

            r = QualityReport()
            for axis in (
                "coverage",
                "faithfulness",
                "diversity",
                "trainability",
                "reward_soundness",
            ):
                r.add(AxisScore(axis, 99.0))
            self.history = [runner_mod.IterationResult(iteration=0, report=r, config_snapshot={})]

        async def run(self) -> object:
            (captured["output_dir"] / "history.json").write_text("[]", encoding="utf-8")  # type: ignore[union-attr]
            return object()

    monkeypatch.setattr(runner_mod, "AutonomousRunner", _FakeRunner)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(tmp_path / "fake-input"),
            "--output",
            str(tmp_path / "out"),
            "--n-trajectories",
            "3",
            "--reward-budget",
            "5",
            "--threshold",
            "90",
            "--max-iterations",
            "1",
        ],
    )

    assert result.exit_code == 0, (
        f"run exited {result.exit_code}; stdout:\n{result.stdout}\nexc: {result.exception!r}"
    )
    assert captured["cfg_threshold"] == 90.0
    assert captured["cfg_max_iter"] == 1
    assert captured["n_trajectories"] == 3
    assert captured["reward_budget"] == 5
    assert (tmp_path / "out" / "config.json").exists()
    assert (tmp_path / "out" / "history.json").exists()


def test_run_subcommand_exits_nonzero_when_axes_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the final report does not cross --threshold, `run` must exit 1."""
    import gyroscope.runner as runner_mod

    class _FailingRunner:
        def __init__(self, cfg, *, threshold: float, max_iterations: int) -> None:
            from gyroscope.quality.metrics import AxisScore, QualityReport

            self._out = cfg.output_dir
            r = QualityReport()
            for axis in (
                "coverage",
                "faithfulness",
                "diversity",
                "trainability",
                "reward_soundness",
            ):
                r.add(AxisScore(axis, 50.0))
            self.history = [runner_mod.IterationResult(iteration=0, report=r, config_snapshot={})]

        async def run(self) -> object:
            self._out.mkdir(parents=True, exist_ok=True)
            (self._out / "history.json").write_text("[]", encoding="utf-8")
            return object()

    monkeypatch.setattr(runner_mod, "AutonomousRunner", _FailingRunner)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(tmp_path / "x"),
            "--output",
            str(tmp_path / "out"),
            "--threshold",
            "95",
            "--max-iterations",
            "0",
        ],
    )

    assert result.exit_code == 1, (
        f"expected exit 1 on failing axes; got {result.exit_code}; stdout:\n{result.stdout}"
    )
