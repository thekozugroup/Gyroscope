"""Targeted tests for the synchronous ``LLMJudge`` rewrite.

These tests verify the two safety properties of the rewritten judge:

1. With no API key (env or explicit) the judge silently falls back to the
   :func:`heuristic_judge` and logs a single warning per instance.
2. When an API key is present, the judge constructs a sync
   ``anthropic.Anthropic`` client and calls ``messages.create`` synchronously
   — no event loop, no async client, no thread-pool tricks.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from gyroscope.rewards.judges import LLMJudge, heuristic_judge

# ---------------------------------------------------------------------------
# 1. Fallback behaviour
# ---------------------------------------------------------------------------


def test_no_api_key_falls_back_to_heuristic_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge = LLMJudge()

    with caplog.at_level("WARNING", logger="gyroscope.rewards.judges"):
        first = judge(["q"], ["alpha beta [KNW-1]"], criterion="alpha beta")
        second = judge(["q"], ["alpha beta [KNW-1]"], criterion="alpha beta")

    expected = heuristic_judge(
        ["q"], ["alpha beta [KNW-1]"], "alpha beta"
    )
    assert first == expected
    assert second == expected

    warnings = [
        r for r in caplog.records if "falling back" in r.getMessage()
    ]
    assert len(warnings) == 1, (
        f"expected exactly one warning per instance, got {len(warnings)}"
    )


def test_each_instance_warns_independently(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level("WARNING", logger="gyroscope.rewards.judges"):
        LLMJudge()(["q"], ["x"], criterion="y")
        LLMJudge()(["q"], ["x"], criterion="y")
    warnings = [
        r for r in caplog.records if "falling back" in r.getMessage()
    ]
    assert len(warnings) == 2


# ---------------------------------------------------------------------------
# 2. Sync client construction
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


class _RecordingMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Message:
        self.calls.append(kwargs)
        return _Message('{"score": 0.42}')


class _RecordingAnthropic:
    instances: ClassVar[list[_RecordingAnthropic]] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.messages = _RecordingMessages()
        _RecordingAnthropic.instances.append(self)


def test_sync_anthropic_client_is_used_when_api_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingAnthropic.instances.clear()
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _RecordingAnthropic)

    judge = LLMJudge(api_key="sk-unit-test", model="claude-judge-stub")
    result = judge(["a prompt"], ["a response"], criterion="be safe")

    assert result == [pytest.approx(0.42)]
    assert len(_RecordingAnthropic.instances) == 1
    client = _RecordingAnthropic.instances[0]
    assert client.api_key == "sk-unit-test"
    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    assert call["model"] == "claude-judge-stub"
    assert call["temperature"] == 0.0
    user_text = call["messages"][0]["content"]
    assert "a response" in user_text
    # Criterion now travels through the cached system block, not the user payload.
    sys_blocks = call["system"]
    assert isinstance(sys_blocks, list)
    assert "be safe" in sys_blocks[0]["text"]
    assert sys_blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_sync_client_is_built_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingAnthropic.instances.clear()
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _RecordingAnthropic)

    judge = LLMJudge(api_key="sk-unit-test")
    assert _RecordingAnthropic.instances == []  # not yet constructed
    judge(["a"], ["b"], criterion="c")
    assert len(_RecordingAnthropic.instances) == 1
    # A second call reuses the same client (no rebuild).
    judge(["a"], ["b"], criterion="c")
    assert len(_RecordingAnthropic.instances) == 1


def test_sync_call_retries_on_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate-limit-style errors should trigger a small retry loop."""

    class _RateLimit(Exception):
        status_code = 429

    attempts: list[int] = []

    class _FlakyMessages:
        def create(self, **kwargs: Any) -> _Message:
            attempts.append(1)
            if len(attempts) < 2:
                raise _RateLimit("slow down")
            return _Message('{"score": 0.8}')

    class _FlakyAnthropic:
        def __init__(self, api_key: str) -> None:
            self.messages = _FlakyMessages()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FlakyAnthropic)
    # Make backoff effectively zero so the test is fast.
    monkeypatch.setattr(
        "gyroscope.rewards.judges._BACKOFF_BASE", 0.0, raising=True
    )

    judge = LLMJudge(api_key="sk-unit-test")
    out = judge(["p"], ["c"], criterion="crit")
    assert out == [pytest.approx(0.8)]
    assert len(attempts) == 2
