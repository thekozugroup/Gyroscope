"""Tests for the AutonomousRunner iteration loop.

We mock every phase, so the runner is exercised in isolation: we control
exactly what each phase returns, then assert (a) the iteration loop only
fires the failing phases on retry, (b) the loop terminates once all axes
pass, and (c) the report/history files land on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gyroscope.core.config import GyroscopeConfig
from gyroscope.core.models import (
    Document,
    DocumentKind,
    GoldenDocument,
    Identity,
    Principle,
    Procedure,
    ProcedureStep,
    RewardKind,
    RewardSpec,
    Trajectory,
    TrajectoryMessage,
)
from gyroscope.runner import AutonomousRunner


def _make_golden(n_principles: int = 30, n_procedures: int = 15) -> GoldenDocument:
    from gyroscope.core.models import KnowledgeItem

    return GoldenDocument(
        identity=Identity(role="X", description="d", mission="m"),
        principles=[
            Principle(id=f"PRN-{i:04d}", statement=f"p{i}", source_chunk_ids=["c1"])
            for i in range(1, n_principles + 1)
        ],
        procedures=[
            Procedure(
                id=f"PRC-{i:04d}",
                name=f"proc{i}",
                purpose="p",
                steps=[ProcedureStep(order=1, action="measure")],
            )
            for i in range(1, n_procedures + 1)
        ],
        knowledge=[
            KnowledgeItem(id=f"KNW-{i:04d}", statement=f"k{i}", citations=["c1"])
            for i in range(1, 101)
        ],
    )


def _traj(idx: int, procedure_id: str, *, golden_text: str = "") -> Trajectory:
    """Build a trajectory whose assistant content cites golden vocabulary so
    the faithfulness metric registers material overlap with the golden doc."""
    citation_padding = " ".join(golden_text.split()[:60]) if golden_text else ""
    return Trajectory(
        id=f"TRJ-{idx:04d}",
        scenario_id=f"SCN-{idx:04d}",
        system="system measure quantities",
        messages=[
            TrajectoryMessage(role="user", content=f"question number {idx}"),
            TrajectoryMessage(
                role="assistant",
                content=(
                    f"answer {idx}: measure quantities for principle p{idx} k{idx}. "
                    f"{citation_padding}"
                ),
            ),
        ],
        tags={
            "procedure_ids": [procedure_id],
            "persona": f"per-{idx % 5}",
            "difficulty": "medium",
        },
    )


def _good_rewards() -> list[RewardSpec]:
    return [
        RewardSpec(
            name="r_safety", kind=RewardKind.SAFETY, description="x", weight=1.0
        ),
        RewardSpec(
            name="r_principle",
            kind=RewardKind.PRINCIPLE,
            description="x",
            weight=1.0,
            principle_ids=["PRN-0001"],
        ),
        RewardSpec(
            name="r_procedure",
            kind=RewardKind.PROCEDURE,
            description="x",
            weight=1.0,
            procedure_ids=["PRC-0001"],
        ),
        RewardSpec(
            name="r_format",
            kind=RewardKind.FORMAT,
            description="x",
            weight=1.0,
            config={"sections": ["Findings"]},
        ),
    ]


@pytest.mark.asyncio
async def test_runner_passes_first_iteration_when_axes_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = GyroscopeConfig(
        input_paths=[tmp_path / "x.txt"], output_dir=tmp_path / "run", api_key="test"
    )
    runner = AutonomousRunner(cfg, threshold=80.0, max_iterations=3)

    golden = _make_golden()
    gtext = golden.to_markdown()
    # Source text mirrors the golden so coverage scores high — we are
    # exercising the loop logic, not the metrics here.
    docs = [Document(source="x.txt", kind=DocumentKind.TXT, text=gtext)]
    train = [_traj(i, f"PRC-{i:04d}", golden_text=gtext) for i in range(1, 11)]
    eval_ = [_traj(99, "PRC-0099")]
    rewards = _good_rewards()

    async def fake_ingest(self, client):  # noqa: ANN001, ARG001
        return docs

    async def fake_curate(self, documents, client):  # noqa: ANN001, ARG001
        return golden

    async def fake_sft(self, golden, client):  # noqa: ANN001, ARG001
        return train, eval_

    async def fake_rewards(self, golden, client):  # noqa: ANN001, ARG001
        return rewards

    class FakeLLM:
        def __init__(self, cfg):
            self.cfg = cfg

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(AutonomousRunner, "_phase_ingest", fake_ingest)
    monkeypatch.setattr(AutonomousRunner, "_phase_curate", fake_curate)
    monkeypatch.setattr(AutonomousRunner, "_phase_sft", fake_sft)
    monkeypatch.setattr(AutonomousRunner, "_phase_rewards", fake_rewards)
    monkeypatch.setattr("gyroscope.runner.LLMClient", FakeLLM)

    artefacts = await runner.run()
    assert artefacts.golden is golden
    assert len(runner.history) == 1
    assert (cfg.output_dir / "report.md").exists()
    assert (cfg.output_dir / "report.html").exists()
    assert (cfg.output_dir / "report.json").exists()
    assert (cfg.output_dir / "history.json").exists()


@pytest.mark.asyncio
async def test_runner_retries_failing_phase_then_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = GyroscopeConfig(
        input_paths=[tmp_path / "x.txt"], output_dir=tmp_path / "run", api_key="test"
    )
    runner = AutonomousRunner(cfg, threshold=70.0, max_iterations=3)

    golden = _make_golden()
    gtext = golden.to_markdown()
    docs = [Document(source="x.txt", kind=DocumentKind.TXT, text=gtext)]
    good_train = [_traj(i, f"PRC-{i:04d}", golden_text=gtext) for i in range(1, 11)]
    # bad train = all identical content -> diversity will tank
    bad = _traj(0, "PRC-0001", golden_text=gtext)
    bad_train = [bad.model_copy(update={"id": f"TRJ-{i:04d}"}) for i in range(10)]

    call_state = {"sft_calls": 0, "rewards_calls": 0, "curate_calls": 0}

    async def fake_ingest(self, client):  # noqa: ANN001, ARG001
        return docs

    async def fake_curate(self, documents, client):  # noqa: ANN001, ARG001
        call_state["curate_calls"] += 1
        return golden

    async def fake_sft(self, golden, client):  # noqa: ANN001, ARG001
        call_state["sft_calls"] += 1
        if call_state["sft_calls"] == 1:
            return bad_train, [_traj(99, "PRC-0099")]
        return good_train, [_traj(99, "PRC-0099")]

    async def fake_rewards(self, golden, client):  # noqa: ANN001, ARG001
        call_state["rewards_calls"] += 1
        return _good_rewards()

    class FakeLLM:
        def __init__(self, cfg):
            self.cfg = cfg

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(AutonomousRunner, "_phase_ingest", fake_ingest)
    monkeypatch.setattr(AutonomousRunner, "_phase_curate", fake_curate)
    monkeypatch.setattr(AutonomousRunner, "_phase_sft", fake_sft)
    monkeypatch.setattr(AutonomousRunner, "_phase_rewards", fake_rewards)
    monkeypatch.setattr("gyroscope.runner.LLMClient", FakeLLM)

    await runner.run()

    # curate ran exactly once, sft ran at least twice (initial + 1 retry triggered by diversity)
    assert call_state["curate_calls"] == 1
    assert call_state["sft_calls"] >= 2

    # --- Stronger assertions: which axis triggered which retry. ---
    # The initial iteration is recorded with no phases_re_run.
    assert runner.history[0].phases_re_run == [], (
        f"iteration 0 should record no retries; got {runner.history[0].phases_re_run!r}"
    )
    # The first retry was triggered by the diversity axis (bad_train is 10
    # near-identical trajectories) — diversity maps to the "sft" phase.
    assert runner.history[1].phases_re_run == ["sft"], (
        f"first retry should be sft only; got {runner.history[1].phases_re_run!r}"
    )
    # Diversity was indeed the failing axis on iteration 0.
    assert "diversity" in runner.history[0].report.axes
    assert runner.history[0].report.axes["diversity"].score < runner.threshold, (
        f"diversity should fail on iter 0 (< {runner.threshold}); got "
        f"{runner.history[0].report.axes['diversity'].score:.2f}"
    )
    # The final iteration crossed the threshold on every axis.
    assert runner.history[-1].report.all_pass(runner.threshold), (
        "expected all axes >= threshold on the final iteration; got "
        + ", ".join(
            f"{n}={a.score:.1f}" for n, a in runner.history[-1].report.axes.items()
        )
    )


@pytest.mark.asyncio
async def test_runner_stops_at_max_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = GyroscopeConfig(
        input_paths=[tmp_path / "x.txt"], output_dir=tmp_path / "run", api_key="test"
    )
    runner = AutonomousRunner(cfg, threshold=99.0, max_iterations=2)

    docs = [Document(source="x.txt", kind=DocumentKind.TXT, text="hello")]
    bad_golden = GoldenDocument(
        identity=Identity(role="X", description="d", mission="m"),
        principles=[Principle(id="PRN-0001", statement="x", source_chunk_ids=[])],
        knowledge=[],
    )

    async def fake_ingest(self, client):  # noqa: ANN001, ARG001
        return docs

    async def fake_curate(self, documents, client):  # noqa: ANN001, ARG001
        return bad_golden

    async def fake_sft(self, golden, client):  # noqa: ANN001, ARG001
        bad = _traj(0, "PRC-0001")
        return [bad.model_copy(update={"id": f"TRJ-{i:04d}"}) for i in range(5)], []

    async def fake_rewards(self, golden, client):  # noqa: ANN001, ARG001
        return [
            RewardSpec(name="r1", kind=RewardKind.LEXICAL, description="x", config={})
        ]

    class FakeLLM:
        def __init__(self, cfg):
            self.cfg = cfg

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(AutonomousRunner, "_phase_ingest", fake_ingest)
    monkeypatch.setattr(AutonomousRunner, "_phase_curate", fake_curate)
    monkeypatch.setattr(AutonomousRunner, "_phase_sft", fake_sft)
    monkeypatch.setattr(AutonomousRunner, "_phase_rewards", fake_rewards)
    monkeypatch.setattr("gyroscope.runner.LLMClient", FakeLLM)

    await runner.run()
    # initial + max_iterations attempts
    assert len(runner.history) == 1 + runner.max_iterations
