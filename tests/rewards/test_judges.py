"""Tests for the heuristic-judge fallback and the LLM judge wiring.

These tests stay offline: they never construct a real Anthropic client and
never hit the network.
"""

from __future__ import annotations

from typing import Any

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
# LLMJudge — pure-sync wiring; the Anthropic client is stubbed.
# ---------------------------------------------------------------------------


class _StubMessage:
    def __init__(self, text: str) -> None:
        self.content = [type("Block", (), {"text": text})()]


class _StubMessages:
    def __init__(self, payloads: list[str]) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _StubMessage:
        self.calls.append(kwargs)
        payload = self._payloads.pop(0) if self._payloads else '{"score": 0.0}'
        return _StubMessage(payload)


class _StubAnthropic:
    def __init__(self, *, api_key: str, payloads: list[str]) -> None:
        self.api_key = api_key
        self.messages = _StubMessages(payloads)


def _install_stub(
    monkeypatch: pytest.MonkeyPatch, payloads: list[str]
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def factory(api_key: str) -> _StubAnthropic:
        stub = _StubAnthropic(api_key=api_key, payloads=payloads)
        captured["client"] = stub
        return stub

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: factory(api_key))
    return captured


class TestLLMJudgeWithStubbedClient:
    def test_invokes_sync_messages_create_with_expected_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_stub(
            monkeypatch, ['{"score": 0.9}', '{"score": 0.1}']
        )
        judge = LLMJudge(api_key="sk-test", model="claude-test")

        result = judge(["p1", "p2"], ["c1", "c2"], criterion="be helpful")

        assert result == [pytest.approx(0.9), pytest.approx(0.1)]
        calls = captured["client"].messages.calls
        assert len(calls) == 2
        assert calls[0]["model"] == "claude-test"
        assert calls[0]["temperature"] == 0.0
        # Criterion now lives in the cached system block, not the user payload.
        sys0 = calls[0]["system"]
        assert isinstance(sys0, list)
        assert "be helpful" in sys0[0]["text"]
        assert sys0[0]["cache_control"] == {"type": "ephemeral"}
        assert "c1" in calls[0]["messages"][0]["content"]
        assert "c2" in calls[1]["messages"][0]["content"]

    def test_falls_back_to_heuristic_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        judge = LLMJudge()

        with caplog.at_level("WARNING", logger="gyroscope.rewards.judges"):
            result = judge(
                ["q"], ["alpha beta [KNW-1]"], criterion="alpha beta"
            )
        # Heuristic = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        assert result == [pytest.approx(1.0)]
        warnings = [
            r for r in caplog.records if "falling back" in r.getMessage()
        ]
        assert len(warnings) == 1

    def test_custom_fallback_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        sentinel = [0.5, 0.5]

        def fallback(
            prompts: Any, completions: Any, *, criterion: str
        ) -> list[float]:
            return sentinel

        judge = LLMJudge(fallback=fallback)
        out = judge(["a", "b"], ["x", "y"], criterion="z")
        assert out == sentinel

    def test_empty_completions_short_circuits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_stub(monkeypatch, [])
        judge = LLMJudge(api_key="sk-test")
        result = judge([], [], criterion="anything")
        assert result == []
        assert "client" not in captured  # client never built

    def test_pads_short_score_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two completions but only one parseable response — the second
        # call returns an unusable payload (we still consider it a success
        # because the client responded, score parses to 0.0).
        _install_stub(monkeypatch, ['{"score": 0.6}', "garbage"])
        judge = LLMJudge(api_key="sk-test")
        result = judge(["a", "b"], ["c", "d"], criterion="crit")
        assert result == [pytest.approx(0.6), 0.0]

    def test_resolves_model_from_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _install_stub(monkeypatch, ['{"score": 0.4}'])

        class _Rewards:
            judge_model = None

        class _Llm:
            judge_model = "fallback-model"

        class _Cfg:
            rewards = _Rewards()
            llm = _Llm()
            api_key = "sk-test"

        judge = LLMJudge(config=_Cfg())
        judge(["a"], ["b"], criterion="c")
        assert captured["client"].messages.calls[0]["model"] == "fallback-model"
