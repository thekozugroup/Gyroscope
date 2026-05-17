"""Prompt-caching guarantees for :class:`LLMJudge`.

The judge system prompt is the only piece of text that is byte-identical
across every ``(prompt, completion)`` pair scored against the same
``criterion``. Routing it through Anthropic's ephemeral prompt cache keeps
the per-call cost flat across a batch and across a full training run.

These tests assert the shape of the request — they never hit the network.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from gyroscope.rewards.judges import LLMJudge


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
        return _Message('{"score": 0.5}')


class _RecordingAnthropic:
    instances: ClassVar[list[_RecordingAnthropic]] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.messages = _RecordingMessages()
        _RecordingAnthropic.instances.append(self)


@pytest.fixture
def patched_anthropic(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingAnthropic]:
    _RecordingAnthropic.instances.clear()
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _RecordingAnthropic)
    return _RecordingAnthropic


def test_judge_system_argument_is_cached_block_list(
    patched_anthropic: type[_RecordingAnthropic],
) -> None:
    """``system`` must be a list of text blocks with ``cache_control`` set,
    not a bare string."""
    judge = LLMJudge(api_key="sk-test", model="claude-judge-stub")

    judge(["p1"], ["c1"], criterion="be helpful")

    client = patched_anthropic.instances[-1]
    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    system_arg = call["system"]

    assert isinstance(system_arg, list), (
        "judge must pass system as a list of typed blocks so prompt caching applies"
    )
    assert len(system_arg) == 1
    block = system_arg[0]
    assert block["type"] == "text"
    assert "be helpful" in block["text"]
    assert block["cache_control"] == {"type": "ephemeral"}


def test_judge_system_block_is_identical_across_pairs(
    patched_anthropic: type[_RecordingAnthropic],
) -> None:
    """The cache only hits if the system block is byte-identical across calls
    sharing the same criterion."""
    judge = LLMJudge(api_key="sk-test", model="claude-judge-stub")

    judge(["p1", "p2", "p3"], ["c1", "c2", "c3"], criterion="be helpful")

    client = patched_anthropic.instances[-1]
    assert len(client.messages.calls) == 3
    system_blocks = [call["system"] for call in client.messages.calls]
    first = system_blocks[0]
    for other in system_blocks[1:]:
        assert other == first


def test_judge_user_payload_is_lightweight_pair_only(
    patched_anthropic: type[_RecordingAnthropic],
) -> None:
    """The per-pair user payload must NOT repeat the criterion — that lives in
    the cached system block now."""
    judge = LLMJudge(api_key="sk-test", model="claude-judge-stub")

    judge(["my prompt"], ["my completion"], criterion="the static principle")

    client = patched_anthropic.instances[-1]
    user_content = client.messages.calls[0]["messages"][0]["content"]
    assert "my prompt" in user_content
    assert "my completion" in user_content
    # The criterion lives only in the cached system block, not duplicated in
    # the per-pair user payload (the whole point of the caching refactor).
    assert "the static principle" not in user_content
