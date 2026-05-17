"""Tests for the streaming swarm (`stream_swarm`).

These cover the round-4 perf refactor:

* surviving trajectories are yielded inline (quality filter is applied before
  the value reaches the consumer);
* a planted duplicate is dropped before the second copy is yielded;
* the in-flight worker count is bounded by ``cfg.llm.max_concurrent`` (or the
  scenario count, whichever is smaller);
* a single worker raising does NOT cancel its peers — the rest of the
  trajectories must still stream out.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass
from typing import Any

import pytest

from gyroscope.core.config import GyroscopeConfig, LLMConfig, SFTConfig
from gyroscope.core.models import Scenario, Trajectory, TrajectoryMessage
from gyroscope.sft import swarm as swarm_mod
from gyroscope.sft.swarm import run_swarm, stream_swarm

from .conftest import FakeLLM, make_golden

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _ClientWithConfig:
    """Wraps :class:`FakeLLM` so it exposes the ``config.llm.max_concurrent``
    knob the streaming swarm uses to size its worker pool."""

    inner: FakeLLM
    max_concurrent: int

    @property
    def config(self) -> GyroscopeConfig:
        return GyroscopeConfig(llm=LLMConfig(max_concurrent=self.max_concurrent), api_key="x")

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)


def _make_scenario(sid: str, *, procedure_id: str | None, seed: str = "default seed") -> Scenario:
    return Scenario(
        id=sid,
        procedure_id=procedure_id,
        principle_ids=["PRN-0001"],
        persona_id="PER-0001",
        difficulty="medium",
        prompt_seed=seed,
    )


def _make_trajectory(scenario: Scenario, *, user: str, score: float) -> Trajectory:
    return Trajectory(
        id=f"TRJ-{scenario.id}",
        scenario_id=scenario.id,
        system="SYS",
        messages=[
            TrajectoryMessage(role="user", content=user),
            TrajectoryMessage(role="assistant", content=f"answer-{scenario.id}"),
        ],
        tags={
            "procedure_ids": [scenario.procedure_id] if scenario.procedure_id else [],
            "principle_ids": list(scenario.principle_ids),
            "persona": scenario.persona_id,
            "difficulty": scenario.difficulty,
        },
        quality_score=score,
        critic_notes="ok",
    )


def _persona_payload(n: int) -> list[dict[str, str]]:
    return [
        {
            "name": f"P{i}",
            "description": f"persona {i}",
            "expertise_level": "intermediate",
            "tone": "neutral",
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_swarm_yields_surviving_trajectories_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternating high/low quality builds; only the high-quality ones reach the stream."""
    golden = make_golden(n_procedures=8, n_principles=2)

    seeds = [{"prompt_seed": f"unique-seed-{i}"} for i in range(8)]
    fake = FakeLLM(json_array_responses=[_persona_payload(2)], json_responses=seeds)
    client = _ClientWithConfig(fake, max_concurrent=4)

    score_cycle = itertools.cycle([0.95, 0.10])
    built_ids: list[str] = []

    async def fake_build(scenario, _golden, _client, *, max_turns, personas, config):  # type: ignore[no-untyped-def]
        built_ids.append(scenario.id)
        score = next(score_cycle)
        return _make_trajectory(scenario, user=scenario.prompt_seed, score=score)

    monkeypatch.setattr(swarm_mod, "build_trajectory", fake_build)

    cfg = SFTConfig(
        n_trajectories=8,
        n_personas=2,
        difficulty_mix={"medium": 1.0},
        critic_min_score=0.7,
        dedup_threshold=0.99,  # near-1 so unique seeds never collide
        eval_holdout_fraction=0.125,  # 1 of 8 procedures
        max_repair_attempts=0,
    )

    streamed: list[tuple[Trajectory, str]] = []
    async for item in stream_swarm(golden, client, cfg):  # type: ignore[arg-type]
        streamed.append(item)

    # Every yielded trajectory must clear the threshold.
    assert streamed, "expected at least one survivor"
    assert all(t.quality_score is not None and t.quality_score >= 0.7 for t, _ in streamed)
    # And the low-quality ones must NOT appear.
    yielded_ids = {t.id for t, _ in streamed}
    assert yielded_ids.issubset({f"TRJ-{sid}" for sid in built_ids})
    # We built 8 with alternating scores → 4 survivors, all unique seeds.
    assert len(streamed) == 4


@pytest.mark.asyncio
async def test_stream_swarm_dedup_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two scenarios with identical seeds produce only one streamed trajectory."""
    golden = make_golden(n_procedures=4, n_principles=2)

    # Plant a duplicate seed at index 1 (same as index 0) — dedup must drop one.
    seeds = [
        {"prompt_seed": "alpha cake recipe please"},
        {"prompt_seed": "alpha cake recipe please"},
        {"prompt_seed": "beta cake recipe please"},
        {"prompt_seed": "gamma cake recipe please"},
    ]
    fake = FakeLLM(json_array_responses=[_persona_payload(2)], json_responses=seeds)
    client = _ClientWithConfig(fake, max_concurrent=1)  # serialise so order is deterministic

    async def fake_build(scenario, _golden, _client, *, max_turns, personas, config):  # type: ignore[no-untyped-def]
        return _make_trajectory(scenario, user=scenario.prompt_seed, score=0.9)

    monkeypatch.setattr(swarm_mod, "build_trajectory", fake_build)

    cfg = SFTConfig(
        n_trajectories=4,
        n_personas=2,
        difficulty_mix={"medium": 1.0},
        critic_min_score=0.5,
        dedup_threshold=0.9,
        eval_holdout_fraction=0.25,
        max_repair_attempts=0,
    )

    streamed: list[Trajectory] = []
    async for traj, _split in stream_swarm(golden, client, cfg):  # type: ignore[arg-type]
        streamed.append(traj)

    user_messages = [traj.messages[0].content for traj in streamed]
    # The duplicate "alpha" seed appears exactly once across the full stream.
    assert user_messages.count("alpha cake recipe please") == 1
    # Three distinct seeds yielded (one alpha + beta + gamma); the second alpha
    # was dropped before it reached the consumer.
    assert len(streamed) == 3


@pytest.mark.asyncio
async def test_stream_swarm_worker_count_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Peak in-flight ``build_trajectory`` calls is capped at ``max_concurrent``."""
    golden = make_golden(n_procedures=12, n_principles=2)

    seeds = [{"prompt_seed": f"seed-{i}"} for i in range(12)]
    fake = FakeLLM(json_array_responses=[_persona_payload(2)], json_responses=seeds)
    max_concurrent = 3
    client = _ClientWithConfig(fake, max_concurrent=max_concurrent)

    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def fake_build(scenario, _golden, _client, *, max_turns, personas, config):  # type: ignore[no-untyped-def]
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            # Give the scheduler a chance to round-robin: if the bound were
            # broken, more than ``max_concurrent`` coroutines would have
            # incremented ``in_flight`` by the time we return.
            await asyncio.sleep(0.01)
            return _make_trajectory(scenario, user=scenario.prompt_seed, score=0.9)
        finally:
            async with lock:
                in_flight -= 1

    monkeypatch.setattr(swarm_mod, "build_trajectory", fake_build)

    cfg = SFTConfig(
        n_trajectories=12,
        n_personas=2,
        difficulty_mix={"medium": 1.0},
        critic_min_score=0.5,
        dedup_threshold=0.99,
        eval_holdout_fraction=0.125,
        max_repair_attempts=0,
    )

    yielded = 0
    async for _ in stream_swarm(golden, client, cfg):  # type: ignore[arg-type]
        yielded += 1

    assert yielded > 0
    assert peak <= max_concurrent, (
        f"peak in-flight {peak} exceeded configured max_concurrent {max_concurrent}"
    )


@pytest.mark.asyncio
async def test_stream_swarm_failure_in_one_worker_does_not_cancel_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One worker raising must not cancel its peers — survivors still stream out."""
    golden = make_golden(n_procedures=6, n_principles=2)

    seeds = [{"prompt_seed": f"seed-{i}"} for i in range(6)]
    fake = FakeLLM(json_array_responses=[_persona_payload(2)], json_responses=seeds)
    client = _ClientWithConfig(fake, max_concurrent=2)

    poison_seed = "seed-2"
    built: list[str] = []

    async def fake_build(scenario, _golden, _client, *, max_turns, personas, config):  # type: ignore[no-untyped-def]
        built.append(scenario.id)
        if scenario.prompt_seed == poison_seed:
            raise RuntimeError("synthetic worker failure")
        return _make_trajectory(scenario, user=scenario.prompt_seed, score=0.9)

    monkeypatch.setattr(swarm_mod, "build_trajectory", fake_build)

    cfg = SFTConfig(
        n_trajectories=6,
        n_personas=2,
        difficulty_mix={"medium": 1.0},
        critic_min_score=0.5,
        dedup_threshold=0.99,
        eval_holdout_fraction=0.1666,  # 1 of 6
        max_repair_attempts=0,
    )

    streamed: list[Trajectory] = []
    async for traj, _split in stream_swarm(golden, client, cfg):  # type: ignore[arg-type]
        streamed.append(traj)

    # Every non-poison scenario should have been built AND yielded.
    assert len(built) == 6, "every scenario must reach the worker even when one raises"
    yielded_seeds = {t.messages[0].content for t in streamed}
    assert poison_seed not in yielded_seeds, "the failing trajectory must not reach the consumer"
    # All five healthy seeds make it through.
    assert len(streamed) == 5


@pytest.mark.asyncio
async def test_run_swarm_tuple_api_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy ``run_swarm() -> (train, eval)`` shape stays green."""
    golden = make_golden(n_procedures=4, n_principles=2)

    seeds = [{"prompt_seed": f"distinct-{i}"} for i in range(4)]
    fake = FakeLLM(json_array_responses=[_persona_payload(2)], json_responses=seeds)
    client = _ClientWithConfig(fake, max_concurrent=2)

    async def fake_build(scenario, _golden, _client, *, max_turns, personas, config):  # type: ignore[no-untyped-def]
        return _make_trajectory(scenario, user=scenario.prompt_seed, score=0.9)

    monkeypatch.setattr(swarm_mod, "build_trajectory", fake_build)

    cfg = SFTConfig(
        n_trajectories=4,
        n_personas=2,
        difficulty_mix={"medium": 1.0},
        critic_min_score=0.5,
        dedup_threshold=0.99,
        eval_holdout_fraction=0.25,
        max_repair_attempts=0,
    )

    train, evals = await run_swarm(golden, client, cfg)  # type: ignore[arg-type]
    assert isinstance(train, list)
    assert isinstance(evals, list)
    assert len(train) + len(evals) == 4
    train_procs = {pid for t in train for pid in t.tags.get("procedure_ids", [])}
    eval_procs = {pid for t in evals for pid in t.tags.get("procedure_ids", [])}
    assert train_procs.isdisjoint(eval_procs)
