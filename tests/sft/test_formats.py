"""Format writers: round-trip Trajectory → sharegpt/chatml/alpaca structure checks."""

from __future__ import annotations

import pytest

from gyroscope.core.models import Trajectory, TrajectoryMessage
from gyroscope.sft.formats import render, to_alpaca, to_chatml, to_sharegpt

from .conftest import make_trajectory


def test_sharegpt_includes_system_and_role_translation():
    traj = make_trajectory(
        system="SYS PROMPT",
        user_text="hi",
        assistant_text="hello back",
        tags={"persona": "PER-0001", "difficulty": "easy", "procedure_ids": ["PRC-0001"]},
    )
    row = to_sharegpt(traj)

    assert row["id"] == traj.id
    assert row["scenario_id"] == traj.scenario_id
    assert row["quality_score"] == traj.quality_score
    assert row["tags"]["difficulty"] == "easy"

    convs = row["conversations"]
    assert convs[0] == {"from": "system", "value": "SYS PROMPT"}
    assert convs[1] == {"from": "human", "value": "hi"}
    assert convs[2] == {"from": "gpt", "value": "hello back"}


def test_sharegpt_tool_role_carries_name():
    traj = Trajectory(
        id="TRJ-0001",
        scenario_id="SCN-0001",
        system="S",
        messages=[
            TrajectoryMessage(role="user", content="run a search"),
            TrajectoryMessage(role="tool", name="search", content="results..."),
            TrajectoryMessage(role="assistant", content="here is your answer"),
        ],
        tags={"persona": "PER-0001", "difficulty": "easy"},
        quality_score=0.8,
    )
    row = to_sharegpt(traj)
    tool_entry = next(c for c in row["conversations"] if c["from"] == "tool")
    assert tool_entry["name"] == "search"
    assert tool_entry["value"] == "results..."


def test_chatml_preserves_native_roles():
    traj = make_trajectory(system="SYS", user_text="q", assistant_text="a")
    row = to_chatml(traj)
    msgs = row["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "q"}
    assert msgs[2] == {"role": "assistant", "content": "a"}
    assert row["tags"] == traj.tags


def test_alpaca_collapses_multi_turn():
    traj = Trajectory(
        id="TRJ-1",
        scenario_id="SCN-1",
        system="SYSTEM_PROMPT",
        messages=[
            TrajectoryMessage(role="user", content="first user"),
            TrajectoryMessage(role="assistant", content="first assist"),
            TrajectoryMessage(role="user", content="final user"),
            TrajectoryMessage(role="assistant", content="final assist"),
        ],
        tags={"persona": "PER-0001", "difficulty": "medium"},
        quality_score=0.91,
    )
    row = to_alpaca(traj)
    assert row["instruction"] == "final user"
    assert row["output"] == "final assist"
    # input should contain the system prompt and the prior turns.
    assert "SYSTEM_PROMPT" in row["input"]
    assert "first user" in row["input"]
    assert "first assist" in row["input"]
    # final assistant must NOT appear in input.
    assert "final assist" not in row["input"]


def test_alpaca_requires_user_and_assistant():
    bad = Trajectory(
        id="TRJ-2",
        scenario_id="SCN-2",
        system="S",
        messages=[TrajectoryMessage(role="assistant", content="lonely")],
        tags={},
    )
    with pytest.raises(ValueError):
        to_alpaca(bad)


def test_render_dispatch():
    traj = make_trajectory()
    assert render(traj, "sharegpt") == to_sharegpt(traj)
    assert render(traj, "chatml") == to_chatml(traj)
    assert render(traj, "alpaca") == to_alpaca(traj)
    with pytest.raises(ValueError):
        render(traj, "unknown-format")


def test_sharegpt_does_not_double_system_when_already_present():
    traj = Trajectory(
        id="TRJ-3",
        scenario_id="SCN-3",
        system="OUTER SYS",
        messages=[
            TrajectoryMessage(role="system", content="INNER SYS"),
            TrajectoryMessage(role="user", content="hi"),
            TrajectoryMessage(role="assistant", content="hello"),
        ],
        tags={},
    )
    row = to_sharegpt(traj)
    system_entries = [c for c in row["conversations"] if c["from"] == "system"]
    assert len(system_entries) == 1
    assert system_entries[0]["value"] == "INNER SYS"
