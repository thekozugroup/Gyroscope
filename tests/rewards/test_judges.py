"""Tests for the LLM judge adapter and the heuristic fallback."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gyroscope.rewards.judges import LLMJudge, heuristic_judge

# ---------------------------------------------------------------------------
# heuristic_judge
# ---------------------------------------------------------------------------


class TestHeuristicJudge:
    def test_full_overlap_with_citation_is_one(self) -> None:
        scores = heuristic_judge(
            ["q"],
            ["alpha beta gamma [KNW-0001]"],
            "alpha beta gamma",
        )
        assert scores == [pytest.approx(1.0)]

    def test_overlap_only(self) -> None:
        scores = heuristic_judge(["q"], ["alpha beta"], "alpha beta")
        # No citation, full overlap -> 0.7
        assert scores == [pytest.approx(0.7)]

    def test_citation_only(self) -> None:
        scores = heuristic_judge(["q"], ["unrelated [KNW-0001]"], "alpha beta")
        # No overlap, has citation -> 0.3
        assert scores == [pytest.approx(0.3)]

    def test_empty_completion_is_zero(self) -> None:
        scores = heuristic_judge(["q"], [""], "alpha")
        assert scores == [0.0]

    def test_deterministic_repeated_calls(self) -> None:
        a = heuristic_judge(["x"], ["alpha beta [KNW-1]"], "alpha gamma")
        b = heuristic_judge(["x"], ["alpha beta [KNW-1]"], "alpha gamma")
        assert a == b

    def test_clips_to_unit_interval(self) -> None:
        scores = heuristic_judge(
            ["q"] * 3,
            ["alpha [KNW-1]", "", "alpha beta gamma [KNW-1]"],
            "alpha beta",
        )
        for s in scores:
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# LLMJudge — uses a mocked async client
# ---------------------------------------------------------------------------


def _make_mock_client(
    *, scores: list[float] | None = None, exception: Exception | None = None
) -> tuple[MagicMock, AsyncMock]:
    client = MagicMock()
    complete_json = AsyncMock()
    if exception is not None:
        complete_json.side_effect = exception
    else:
        complete_json.return_value = {"scores": scores or []}
    client.complete_json = complete_json
    return client, complete_json


def _make_config(judge_model: str | None = "claude-test") -> Any:
    cfg = MagicMock()
    cfg.rewards.judge_model = judge_model
    cfg.llm.judge_model = "fallback-model"
    return cfg


class TestLLMJudge:
    def test_invokes_client_with_expected_args(self) -> None:
        client, complete_json = _make_mock_client(scores=[0.9, 0.1])
        cfg = _make_config()
        judge = LLMJudge(client=client, config=cfg)

        result = judge(["p1", "p2"], ["c1", "c2"], "be helpful")

        assert result == [pytest.approx(0.9), pytest.approx(0.1)]
        complete_json.assert_awaited_once()
        kwargs = complete_json.await_args.kwargs
        assert kwargs["model"] == "claude-test"
        assert "be helpful" in kwargs["user"]
        assert "c1" in kwargs["user"] and "c2" in kwargs["user"]
        assert kwargs["temperature"] == 0.0

    def test_falls_back_to_heuristic_on_client_error(self) -> None:
        client, _ = _make_mock_client(exception=RuntimeError("nope"))
        cfg = _make_config()
        judge = LLMJudge(client=client, config=cfg)

        result = judge(["q"], ["alpha beta [KNW-1]"], "alpha beta")
        # Heuristic = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert result == [pytest.approx(1.0)]

    def test_custom_fallback_used(self) -> None:
        client, _ = _make_mock_client(exception=RuntimeError("nope"))
        cfg = _make_config()
        sentinel = [0.5, 0.5]

        def fallback(prompts, completions, criterion):
            return sentinel

        judge = LLMJudge(client=client, config=cfg, fallback=fallback)
        out = judge(["a", "b"], ["x", "y"], "z")
        assert out == sentinel

    def test_empty_completions_short_circuits(self) -> None:
        client, complete_json = _make_mock_client(scores=[])
        cfg = _make_config()
        judge = LLMJudge(client=client, config=cfg)
        result = judge([], [], "anything")
        assert result == []
        complete_json.assert_not_awaited()

    def test_pads_short_score_list(self) -> None:
        client, _ = _make_mock_client(scores=[0.6])
        cfg = _make_config()
        judge = LLMJudge(client=client, config=cfg)
        result = judge(["a", "b"], ["c", "d"], "crit")
        assert result == [pytest.approx(0.6), 0.0]

    def test_uses_llm_config_judge_model_when_rewards_unset(self) -> None:
        client, complete_json = _make_mock_client(scores=[0.4])
        cfg = _make_config(judge_model=None)
        judge = LLMJudge(client=client, config=cfg)
        judge(["a"], ["b"], "c")
        assert complete_json.await_args.kwargs["model"] == "fallback-model"
