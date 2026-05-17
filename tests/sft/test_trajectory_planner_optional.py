"""When SFTConfig.use_planner=False the planner LLM step must be skipped.

The deterministic scenario generator already picks principle_ids and
procedure_id, so the planner is pure overhead in steady-state runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from gyroscope.core.config import SFTConfig
from gyroscope.core.models import Scenario
from gyroscope.sft import trajectory as traj_mod
from gyroscope.sft.trajectory import build_trajectory

from .conftest import make_golden


def _scenario() -> Scenario:
    return Scenario(
        id="SCN-0001",
        procedure_id="PRC-0001",
        principle_ids=["PRN-0001", "PRN-0002"],
        persona_id="PER-0001",
        difficulty="medium",
        prompt_seed="explain please",
    )


@pytest.mark.asyncio
async def test_planner_skipped_when_use_planner_false(monkeypatch):
    golden = make_golden()
    scenario = _scenario()

    counts = {"planner": 0, "user": 0, "assistant": 0, "critic": 0}

    async def fake_planner(scen, gold, client, config=None):
        counts["planner"] += 1
        return {
            "principle_ids": [],
            "procedure_id": None,
            "outline": ["x"],
            "max_turns": 1,
        }

    async def fake_user(**kwargs: Any) -> str:
        counts["user"] += 1
        return "hi"

    async def fake_assistant(**kwargs: Any) -> str:
        counts["assistant"] += 1
        return "hello"

    async def fake_critic(trj, gold, client, **kwargs: Any):
        counts["critic"] += 1
        return 0.9, "ok"

    monkeypatch.setattr(traj_mod, "_planner_step", fake_planner)
    monkeypatch.setattr(traj_mod, "_user_sim_turn", fake_user)
    monkeypatch.setattr(traj_mod, "_assistant_turn", fake_assistant)
    monkeypatch.setattr(traj_mod, "_critic_score", fake_critic)

    cfg = SFTConfig(use_planner=False, max_turns=1)
    t = await build_trajectory(
        scenario,
        golden,
        client=object(),
        config=cfg,  # type: ignore[arg-type]
    )

    # The planner LLM step must NOT have been called.
    assert counts["planner"] == 0
    # The other agents still ran.
    assert counts["user"] == 1
    assert counts["assistant"] == 1
    assert counts["critic"] == 1

    # The scenario-derived selections survive: tags reflect the scenario,
    # not whatever the (uncalled) planner might have produced.
    assert t.tags["principle_ids"] == ["PRN-0001", "PRN-0002"]
    assert t.tags["procedure_ids"] == ["PRC-0001"]


@pytest.mark.asyncio
async def test_planner_called_when_use_planner_true(monkeypatch):
    """Sanity check: the planner IS called when the flag is on."""
    golden = make_golden()
    scenario = _scenario()

    counts = {"planner": 0}

    async def fake_planner(scen, gold, client, config=None):
        counts["planner"] += 1
        return {
            "principle_ids": ["PRN-0001"],
            "procedure_id": None,
            "outline": ["x"],
            "max_turns": 1,
        }

    async def fake_user(**kwargs: Any) -> str:
        return "hi"

    async def fake_assistant(**kwargs: Any) -> str:
        return "hello"

    async def fake_critic(trj, gold, client, **kwargs: Any):
        return 0.95, "ok"

    monkeypatch.setattr(traj_mod, "_planner_step", fake_planner)
    monkeypatch.setattr(traj_mod, "_user_sim_turn", fake_user)
    monkeypatch.setattr(traj_mod, "_assistant_turn", fake_assistant)
    monkeypatch.setattr(traj_mod, "_critic_score", fake_critic)

    cfg = SFTConfig(use_planner=True, max_turns=1)
    await build_trajectory(
        scenario,
        golden,
        client=object(),
        config=cfg,  # type: ignore[arg-type]
    )

    assert counts["planner"] == 1
