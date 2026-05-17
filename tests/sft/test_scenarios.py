"""Scenario generator tests covering stratification and round-robin coverage."""

from __future__ import annotations

from collections import Counter

import pytest

from gyroscope.core.models import Persona
from gyroscope.sft.scenarios import (
    build_anchors,
    difficulty_targets,
    generate_scenarios,
    round_robin_anchors,
)

from .conftest import FakeLLM, make_golden


def _personas(n: int = 3) -> list[Persona]:
    return [
        Persona(
            id=f"PER-{i:04d}",
            name=f"P{i}",
            description=f"persona {i}",
            expertise_level="intermediate",
            tone="neutral",
        )
        for i in range(1, n + 1)
    ]


def test_difficulty_targets_sum_to_n():
    out = difficulty_targets(20, {"easy": 0.25, "medium": 0.5, "hard": 0.2, "adversarial": 0.05})
    assert sum(out.values()) == 20
    # medium should be the largest bucket
    assert out["medium"] == max(out.values())


def test_difficulty_targets_default_when_weights_zero():
    out = difficulty_targets(10, {"easy": 0, "medium": 0, "hard": 0, "adversarial": 0})
    assert out["medium"] == 10
    assert out["easy"] == 0


def test_round_robin_anchor_coverage_visits_all_before_repeat():
    anchors = build_anchors(make_golden(n_procedures=4))
    seq = round_robin_anchors(anchors, n_total=4)
    procedure_ids = [a[0] for a in seq]
    assert set(procedure_ids) == {"PRC-0001", "PRC-0002", "PRC-0003", "PRC-0004"}


def test_round_robin_when_more_scenarios_than_anchors_cycles():
    anchors = build_anchors(make_golden(n_procedures=2))
    seq = round_robin_anchors(anchors, n_total=5)
    procedure_ids = [a[0] for a in seq]
    # Both procedures show up in the first 2 entries.
    assert {procedure_ids[0], procedure_ids[1]} == {"PRC-0001", "PRC-0002"}
    # And it then cycles.
    assert procedure_ids[2] == procedure_ids[0]


@pytest.mark.asyncio
async def test_generate_scenarios_distribution_roughly_matches_mix():
    golden = make_golden(n_procedures=3)
    personas = _personas(3)
    n_total = 20
    mix = {"easy": 0.25, "medium": 0.5, "hard": 0.2, "adversarial": 0.05}

    # Stub returns the same prompt seed for every call.
    fake = FakeLLM(
        json_responses=[{"prompt_seed": f"seed-{i}"} for i in range(n_total)]
    )
    scenarios = await generate_scenarios(golden, personas, fake, n_total, mix)  # type: ignore[arg-type]

    assert len(scenarios) == n_total
    # ids deterministic
    assert scenarios[0].id == "SCN-0001"
    assert scenarios[-1].id == f"SCN-{n_total:04d}"

    # Distribution is within reason — exact counts come from difficulty_targets.
    expected = difficulty_targets(n_total, mix)
    actual = Counter(s.difficulty for s in scenarios)
    for k, v in expected.items():
        assert actual.get(k, 0) == v

    # Every procedure id covered at least once before any has two.
    procedure_visits: dict[str, int] = {}
    saw_repeat = False
    for s in scenarios:
        if s.procedure_id is None:
            continue
        procedure_visits[s.procedure_id] = procedure_visits.get(s.procedure_id, 0) + 1
        if procedure_visits[s.procedure_id] > 1 and len(procedure_visits) < len(
            golden.procedures
        ):
            saw_repeat = True
    assert not saw_repeat


@pytest.mark.asyncio
async def test_generate_scenarios_falls_back_when_llm_empty():
    golden = make_golden(n_procedures=2)
    personas = _personas(1)
    fake = FakeLLM(json_responses=[{} for _ in range(4)])
    scenarios = await generate_scenarios(
        golden, personas, fake, 4, {"medium": 1.0}  # type: ignore[arg-type]
    )
    assert len(scenarios) == 4
    # Fallback seed mentions persona expertise.
    assert all(s.prompt_seed for s in scenarios)
