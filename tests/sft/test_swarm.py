"""End-to-end swarm tests with a fully mocked LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gyroscope.core.config import SFTConfig
from gyroscope.core.io import read_jsonl
from gyroscope.core.models import Persona, Scenario, Trajectory, TrajectoryMessage
from gyroscope.sft import swarm as swarm_mod
from gyroscope.sft.pipeline import SFTPipeline
from gyroscope.sft.swarm import run_swarm, semantic_dedup

from .conftest import FakeLLM, make_golden


def _persona(idx: int = 1) -> Persona:
    return Persona(
        id=f"PER-{idx:04d}",
        name=f"P{idx}",
        description=f"persona {idx}",
        expertise_level="intermediate",
        tone="neutral",
    )


def _make_scenario(
    sid: str, *, procedure_id: str | None, seed: str = "default seed"
) -> Scenario:
    return Scenario(
        id=sid,
        procedure_id=procedure_id,
        principle_ids=["PRN-0001"],
        persona_id="PER-0001",
        difficulty="medium",
        prompt_seed=seed,
    )


def _make_trajectory(scenario: Scenario, *, user: str, assistant: str, score: float) -> Trajectory:
    return Trajectory(
        id=f"TRJ-{scenario.id}",
        scenario_id=scenario.id,
        system="SYS",
        messages=[
            TrajectoryMessage(role="user", content=user),
            TrajectoryMessage(role="assistant", content=assistant),
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


def test_semantic_dedup_drops_planted_duplicate():
    s1 = _make_scenario("SCN-0001", procedure_id="PRC-0001", seed="how do I cook eggs")
    s2 = _make_scenario("SCN-0002", procedure_id="PRC-0002", seed="how do I cook eggs")
    s3 = _make_scenario("SCN-0003", procedure_id="PRC-0003", seed="explain quantum tunneling")

    t1 = _make_trajectory(s1, user="how do i cook eggs?", assistant="boil them", score=0.9)
    t2 = _make_trajectory(s2, user="how do i cook eggs?", assistant="fry them", score=0.9)
    t3 = _make_trajectory(s3, user="explain quantum tunneling", assistant="...", score=0.9)

    kept = semantic_dedup([t1, t2, t3], {s1.id: s1, s2.id: s2, s3.id: s3}, threshold=0.9)
    assert [t.id for t in kept] == [t1.id, t3.id]


@pytest.mark.asyncio
async def test_run_swarm_is_leak_free_dedupes_and_writes_to_disk(monkeypatch, tmp_path: Path):
    """End-to-end with hand-crafted golden + mocked LLM throughout."""
    golden = make_golden(n_procedures=4, n_principles=3)

    # 1) Personas come from a fake LLM array call.
    fake = FakeLLM(
        json_array_responses=[
            [
                {
                    "name": f"P{i}",
                    "description": f"persona {i}",
                    "expertise_level": "intermediate",
                    "tone": "neutral",
                }
                for i in range(1, 3)
            ]
        ],
        # 2) Scenario gen: one JSON object per scenario call.
        #    With n_trajectories=8 and 4 procedures, round-robin visits each
        #    twice. We'll plant duplicate seeds for two of them so dedup fires.
        json_responses=[
            {"prompt_seed": "alpha cake recipe please"},
            {"prompt_seed": "beta cake recipe please"},
            {"prompt_seed": "gamma cake recipe please"},
            {"prompt_seed": "delta cake recipe please"},
            {"prompt_seed": "alpha cake recipe please"},  # duplicate of scenario 1
            {"prompt_seed": "epsilon cake recipe please"},
            {"prompt_seed": "zeta cake recipe please"},
            {"prompt_seed": "eta cake recipe please"},
        ],
    )

    # 3) Stub build_trajectory so we don't have to mock the four agents here.
    seen_scenarios: list[Scenario] = []

    async def fake_build(
        scenario: Scenario,
        gold: Any,
        client: Any,
        max_turns: int = 6,
        *,
        personas: Any = None,
        config: Any = None,
    ) -> Trajectory:
        seen_scenarios.append(scenario)
        return _make_trajectory(
            scenario,
            user=scenario.prompt_seed,
            assistant=f"answer for {scenario.id}",
            score=0.9,
        )

    monkeypatch.setattr(swarm_mod, "build_trajectory", fake_build)

    cfg = SFTConfig(
        n_trajectories=8,
        n_personas=2,
        difficulty_mix={"medium": 1.0},
        critic_min_score=0.7,
        max_repair_attempts=0,
        dedup_threshold=0.9,
        eval_holdout_fraction=0.25,  # 25% of 4 procedures → 1 eval procedure
    )

    train, evals = await run_swarm(golden, fake, cfg)  # type: ignore[arg-type]

    # --- leak-free at procedure level ---
    train_proc_ids = {pid for t in train for pid in t.tags.get("procedure_ids", [])}
    eval_proc_ids = {pid for t in evals for pid in t.tags.get("procedure_ids", [])}
    assert eval_proc_ids, "eval split should contain at least one trajectory"
    assert train_proc_ids.isdisjoint(eval_proc_ids), (
        f"leakage: {train_proc_ids & eval_proc_ids}"
    )

    # --- dedup removed the duplicate ---
    # Built 8 trajectories total; one pair shares "alpha cake recipe please".
    total_built = len(seen_scenarios)
    total_kept = len(train) + len(evals)
    assert total_built == 8
    assert total_kept == 7, f"expected 7 after dedup, got {total_kept}"

    # --- pipeline writes both files ---
    pipeline = SFTPipeline()

    # Reset fake responses for the second run-through (pipeline calls run_swarm again).
    fake2 = FakeLLM(
        json_array_responses=[
            [
                {
                    "name": f"P{i}",
                    "description": f"persona {i}",
                    "expertise_level": "intermediate",
                    "tone": "neutral",
                }
                for i in range(1, 3)
            ]
        ],
        json_responses=[
            {"prompt_seed": f"unique-seed-{i}"} for i in range(8)
        ],
    )
    train_path, eval_path = await pipeline.run(golden, tmp_path, fake2, cfg)  # type: ignore[arg-type]
    assert train_path.exists() and train_path.name == "sft.jsonl"
    assert eval_path.exists() and eval_path.name == "eval.jsonl"

    train_rows = list(read_jsonl(train_path))
    eval_rows = list(read_jsonl(eval_path))
    assert len(train_rows) >= 1
    assert len(eval_rows) >= 1
    # Default format is ShareGPT — every row has "conversations".
    assert all("conversations" in r for r in train_rows + eval_rows)


@pytest.mark.asyncio
async def test_run_swarm_returns_empty_when_all_below_threshold(monkeypatch) -> None:
    """Every trajectory scores 0.1; critic_min_score=0.95 — both splits empty.

    Pins the contract that ``run_swarm`` quietly drops sub-threshold trajectories
    rather than raising. If a future change starts raising on an all-empty
    result, this test fails so the caller is forced to handle it.
    """
    golden = make_golden(n_procedures=2, n_principles=2)

    fake = FakeLLM(
        json_array_responses=[
            [
                {
                    "name": f"P{i}",
                    "description": f"persona {i}",
                    "expertise_level": "intermediate",
                    "tone": "neutral",
                }
                for i in range(1, 3)
            ]
        ],
        json_responses=[
            {"prompt_seed": f"easy seed {i}"} for i in range(4)
        ],
    )

    async def fake_build(
        scenario: Scenario,
        gold: Any,
        client: Any,
        max_turns: int = 6,
        *,
        personas: Any = None,
        config: Any = None,
    ) -> Trajectory:
        # Every trajectory comes back well below critic_min_score=0.95.
        return _make_trajectory(
            scenario,
            user=scenario.prompt_seed,
            assistant=f"weak answer for {scenario.id}",
            score=0.1,
        )

    monkeypatch.setattr(swarm_mod, "build_trajectory", fake_build)

    cfg = SFTConfig(
        critic_min_score=0.95,
        n_trajectories=4,
        n_personas=2,
        difficulty_mix={"easy": 1.0},
        eval_holdout_fraction=0.5,
        max_repair_attempts=0,
    )

    train, evals = await run_swarm(golden, fake, cfg)  # type: ignore[arg-type]

    assert (train, evals) == ([], []), (
        f"expected ([], []) when every trajectory is sub-threshold, got "
        f"({len(train)} train, {len(evals)} eval)"
    )
