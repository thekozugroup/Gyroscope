"""Tests for the deterministic reward designer."""

from __future__ import annotations

from collections import Counter

import pytest

from gyroscope.core.config import RewardConfig
from gyroscope.core.models import (
    AntiPattern,
    GoldenDocument,
    Identity,
    KnowledgeItem,
    Principle,
    Procedure,
    ProcedureStep,
    RewardKind,
    VocabularyTerm,
)
from gyroscope.rewards.designer import design_rewards, priority_for_kind


def _make_golden(
    *,
    n_principles: int = 3,
    n_procedures: int = 2,
    n_anti_patterns: int = 1,
    n_vocab: int = 2,
    n_knowledge: int = 1,
    principle_extras: list[str] | None = None,
) -> GoldenDocument:
    principles = [
        Principle(
            id=f"PRN-{i:04d}",
            statement=f"Principle {i}: always be precise and cite sources.",
            weight=1.0 + i * 0.1,
        )
        for i in range(n_principles)
    ]
    if principle_extras:
        for j, text in enumerate(principle_extras):
            principles.append(
                Principle(id=f"PRN-9{j:03d}", statement=text, weight=0.5)
            )
    procedures = [
        Procedure(
            id=f"PRC-{i:04d}",
            name=f"procedure_{i}",
            purpose="Perform some structured workflow.",
            steps=[
                ProcedureStep(order=1, action="Gather requirements from stakeholders"),
                ProcedureStep(order=2, action="Design solution with constraints"),
                ProcedureStep(order=3, action="Ship implementation to staging"),
            ],
        )
        for i in range(n_procedures)
    ]
    anti = [
        AntiPattern(
            id=f"ANT-{i:04d}",
            description=f"Never fabricate citations or invent numbers {i}",
            why_bad="Unreliable",
            correction="Cite real sources",
        )
        for i in range(n_anti_patterns)
    ]
    vocab = [
        VocabularyTerm(term=f"term_{i}", definition="definition")
        for i in range(n_vocab)
    ]
    knowledge = [
        KnowledgeItem(id=f"KNW-{i:04d}", statement="fact")
        for i in range(n_knowledge)
    ]
    return GoldenDocument(
        identity=Identity(
            role="Test Role",
            description="A test agent",
            mission="Help with tests",
        ),
        principles=principles,
        procedures=procedures,
        knowledge=knowledge,
        vocabulary=vocab,
        anti_patterns=anti,
    )


@pytest.mark.asyncio
async def test_designer_emits_expected_kinds() -> None:
    golden = _make_golden(
        principle_extras=["responses must include markdown sections"],
    )
    cfg = RewardConfig(reward_budget=20)
    bundle = await design_rewards(golden, None, cfg)

    kinds = {s.kind for s in bundle.specs}
    assert RewardKind.SAFETY in kinds
    assert RewardKind.PROCEDURE in kinds
    assert RewardKind.PRINCIPLE in kinds
    assert RewardKind.CITATION in kinds
    assert RewardKind.FORMAT in kinds
    assert RewardKind.LEXICAL in kinds
    assert RewardKind.LENGTH in kinds

    # Each procedure produces its own spec.
    proc_specs = [s for s in bundle.specs if s.kind is RewardKind.PROCEDURE]
    assert len(proc_specs) == len(golden.procedures)
    for s in proc_specs:
        assert s.procedure_ids
        assert s.config["ordered_steps"]


@pytest.mark.asyncio
async def test_designer_respects_budget_priority() -> None:
    golden = _make_golden(n_principles=8, n_procedures=3, n_anti_patterns=2)
    cfg = RewardConfig(reward_budget=3)
    bundle = await design_rewards(golden, None, cfg)

    assert len(bundle.specs) == 3
    kinds = {s.kind for s in bundle.specs}
    # Safety has top priority; should survive when present.
    assert RewardKind.SAFETY in kinds
    # Procedure (priority 1) outranks length (priority 6), so length should be dropped first.
    assert RewardKind.LENGTH not in kinds
    # Procedure should be present given priority 1.
    assert RewardKind.PROCEDURE in kinds


@pytest.mark.asyncio
async def test_designer_skips_excluded_kinds() -> None:
    golden = _make_golden()
    cfg = RewardConfig(
        reward_budget=20,
        include_kinds=["procedure", "safety"],
    )
    bundle = await design_rewards(golden, None, cfg)
    kinds = {s.kind for s in bundle.specs}
    assert kinds <= {RewardKind.PROCEDURE, RewardKind.SAFETY}
    assert RewardKind.PRINCIPLE not in kinds
    assert RewardKind.LENGTH not in kinds


@pytest.mark.asyncio
async def test_principle_ids_propagated() -> None:
    golden = _make_golden(n_principles=5, n_procedures=0, n_anti_patterns=0, n_vocab=0)
    cfg = RewardConfig(reward_budget=20, include_kinds=["principle"])
    bundle = await design_rewards(golden, None, cfg)

    principle_ids = [pid for s in bundle.specs for pid in s.principle_ids]
    # Highest-weight principles should be selected.
    assert len(principle_ids) == min(5, cfg.reward_budget // 2)
    # All ids appear in the original principles set.
    valid = {p.id for p in golden.principles}
    assert set(principle_ids) <= valid


@pytest.mark.asyncio
async def test_unique_names() -> None:
    golden = _make_golden(n_procedures=3, n_principles=4)
    cfg = RewardConfig(reward_budget=20)
    bundle = await design_rewards(golden, None, cfg)
    names = [s.name for s in bundle.specs]
    counter = Counter(names)
    duplicates = [n for n, c in counter.items() if c > 1]
    assert not duplicates


@pytest.mark.asyncio
async def test_no_anti_patterns_no_safety_reward() -> None:
    golden = _make_golden(n_anti_patterns=0)
    cfg = RewardConfig(reward_budget=20)
    bundle = await design_rewards(golden, None, cfg)
    kinds = {s.kind for s in bundle.specs}
    assert RewardKind.SAFETY not in kinds


@pytest.mark.asyncio
async def test_priority_table_complete() -> None:
    for kind in RewardKind:
        # Must not raise.
        priority_for_kind(kind)


@pytest.mark.asyncio
async def test_golden_role_preserved() -> None:
    golden = _make_golden()
    cfg = RewardConfig(reward_budget=4)
    bundle = await design_rewards(golden, None, cfg)
    assert bundle.golden_role == golden.identity.role
    assert bundle.version == "1"
