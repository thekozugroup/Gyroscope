"""Trajectory orchestrator tests with all four agent steps mocked."""

from __future__ import annotations

from typing import Any

import pytest

from gyroscope.core.config import SFTConfig
from gyroscope.core.models import Persona, Scenario
from gyroscope.sft import trajectory as traj_mod
from gyroscope.sft.trajectory import build_trajectory

from .conftest import make_golden


def _persona() -> Persona:
    return Persona(
        id="PER-0001",
        name="Tester",
        description="A QA persona.",
        expertise_level="intermediate",
        tone="curt",
    )


def _scenario() -> Scenario:
    return Scenario(
        id="SCN-0001",
        procedure_id="PRC-0001",
        principle_ids=["PRN-0001", "PRN-0002"],
        persona_id="PER-0001",
        difficulty="medium",
        prompt_seed="How do I do procedure 1?",
    )


@pytest.mark.asyncio
async def test_trajectory_sequencing_and_critic(monkeypatch):
    """Two user turns, two assistant turns, critic returns 0.9."""
    golden = make_golden()
    scenario = _scenario()
    personas = [_persona()]

    user_replies = iter(["First user msg", "Second user msg"])
    assistant_replies = iter(["First assist", "Second assist"])

    async def fake_planner(scen, gold, client, config=None):
        return {
            "principle_ids": ["PRN-0001"],
            "procedure_id": "PRC-0001",
            "outline": ["respond"],
            "max_turns": 2,
        }

    async def fake_user(**kwargs: Any) -> str:
        return next(user_replies)

    async def fake_assistant(**kwargs: Any) -> str:
        return next(assistant_replies)

    async def fake_critic(trj, gold, client, **kwargs: Any):
        return 0.9, "looks good"

    monkeypatch.setattr(traj_mod, "_planner_step", fake_planner)
    monkeypatch.setattr(traj_mod, "_user_sim_turn", fake_user)
    monkeypatch.setattr(traj_mod, "_assistant_turn", fake_assistant)
    monkeypatch.setattr(traj_mod, "_critic_score", fake_critic)

    # use_planner=True so the patched fake_planner gets to set the principle
    # ids the test asserts on.
    cfg = SFTConfig(use_planner=True, max_turns=4)
    t = await build_trajectory(
        scenario, golden, client=object(), max_turns=4, personas=personas,  # type: ignore[arg-type]
        config=cfg,
    )

    roles = [m.role for m in t.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert t.messages[0].content == "First user msg"
    assert t.messages[1].content == "First assist"
    assert t.messages[2].content == "Second user msg"
    assert t.messages[3].content == "Second assist"

    assert t.quality_score == 0.9
    assert t.critic_notes == "looks good"
    assert t.tags["procedure_ids"] == ["PRC-0001"]
    assert t.tags["principle_ids"] == ["PRN-0001"]
    assert t.tags["persona"] == "PER-0001"
    assert t.tags["difficulty"] == "medium"
    assert t.scenario_id == scenario.id
    assert t.id == "TRJ-SCN-0001"


@pytest.mark.asyncio
async def test_user_end_signal_terminates_conversation(monkeypatch):
    golden = make_golden()
    scenario = _scenario()

    user_replies = iter(["opener", "<END>"])
    assistant_replies = iter(["only assistant turn"])

    async def fake_planner(scen, gold, client, config=None):
        return {
            "principle_ids": [],
            "procedure_id": None,
            "outline": ["x"],
            "max_turns": 6,
        }

    async def fake_user(**kwargs: Any) -> str:
        return next(user_replies)

    async def fake_assistant(**kwargs: Any) -> str:
        return next(assistant_replies)

    async def fake_critic(trj, gold, client, **kwargs: Any):
        return 0.95, "ok"

    monkeypatch.setattr(traj_mod, "_planner_step", fake_planner)
    monkeypatch.setattr(traj_mod, "_user_sim_turn", fake_user)
    monkeypatch.setattr(traj_mod, "_assistant_turn", fake_assistant)
    monkeypatch.setattr(traj_mod, "_critic_score", fake_critic)

    t = await build_trajectory(scenario, golden, client=object(), max_turns=6)  # type: ignore[arg-type]
    roles = [m.role for m in t.messages]
    # User said END on turn 2 → only first user/assistant pair present.
    assert roles == ["user", "assistant"]


@pytest.mark.asyncio
async def test_repair_pass_re_rolls_last_assistant_when_below_threshold(monkeypatch):
    golden = make_golden()
    scenario = _scenario()
    cfg = SFTConfig(critic_min_score=0.7, max_repair_attempts=2)

    user_replies = iter(["q"])

    # First assistant response, then two repair re-rolls.
    assistant_replies = iter(["bad", "still-bad", "finally-good"])

    # First critic call below threshold; second still below; third above.
    critic_scores = iter([(0.3, "bad-1"), (0.5, "bad-2"), (0.85, "good")])

    async def fake_planner(scen, gold, client, config=None):
        return {
            "principle_ids": [],
            "procedure_id": None,
            "outline": ["x"],
            "max_turns": 1,
        }

    async def fake_user(**kwargs: Any) -> str:
        return next(user_replies)

    async def fake_assistant(**kwargs: Any) -> str:
        return next(assistant_replies)

    async def fake_critic(trj, gold, client, **kwargs: Any):
        return next(critic_scores)

    monkeypatch.setattr(traj_mod, "_planner_step", fake_planner)
    monkeypatch.setattr(traj_mod, "_user_sim_turn", fake_user)
    monkeypatch.setattr(traj_mod, "_assistant_turn", fake_assistant)
    monkeypatch.setattr(traj_mod, "_critic_score", fake_critic)

    t = await build_trajectory(
        scenario, golden, client=object(), max_turns=1, config=cfg  # type: ignore[arg-type]
    )
    # Last assistant message should be the final re-roll.
    assert t.messages[-1].role == "assistant"
    assert t.messages[-1].content == "finally-good"
    assert t.quality_score == 0.85
    # Critic notes captures repair history.
    assert "repair#" in (t.critic_notes or "")


@pytest.mark.asyncio
async def test_repair_loop_respects_max_attempts(monkeypatch):
    golden = make_golden()
    scenario = _scenario()
    cfg = SFTConfig(critic_min_score=0.7, max_repair_attempts=1)

    user_replies = iter(["q"])
    assistant_replies = iter(["bad", "still-bad"])
    # Both critic calls below threshold — repair attempt cap is 1.
    critic_scores = iter([(0.1, "n1"), (0.2, "n2")])

    async def fake_planner(scen, gold, client, config=None):
        return {
            "principle_ids": [],
            "procedure_id": None,
            "outline": ["x"],
            "max_turns": 1,
        }

    async def fake_user(**kwargs: Any) -> str:
        return next(user_replies)

    async def fake_assistant(**kwargs: Any) -> str:
        return next(assistant_replies)

    async def fake_critic(trj, gold, client, **kwargs: Any):
        return next(critic_scores)

    monkeypatch.setattr(traj_mod, "_planner_step", fake_planner)
    monkeypatch.setattr(traj_mod, "_user_sim_turn", fake_user)
    monkeypatch.setattr(traj_mod, "_assistant_turn", fake_assistant)
    monkeypatch.setattr(traj_mod, "_critic_score", fake_critic)

    t = await build_trajectory(
        scenario, golden, client=object(), max_turns=1, config=cfg  # type: ignore[arg-type]
    )
    # Should have exactly 1 user + 1 assistant; quality stays below threshold.
    assert sum(1 for m in t.messages if m.role == "assistant") == 1
    assert t.quality_score == 0.2
