"""Tests for the :mod:`gyroscope.core.llm` retry policy and role accessors.

These tests stub the underlying Anthropic ``messages.create`` call so they
exercise:

* ``_is_retryable_llm_error`` — that 429 / connection / timeout errors are
  retried and that 400 / 401 / 403 are NOT.
* ``LLMClient.model_for`` / ``LLMClient.temperature_for`` — the new public
  role-to-config accessors.
* ``LLMClient.config`` — the read-only property.

No real network traffic is generated.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIStatusError,
    BadRequestError,
    RateLimitError,
)

from gyroscope.core.config import GyroscopeConfig, LLMConfig
from gyroscope.core.llm import LLMClient, _is_retryable_llm_error

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> LLMClient:
    cfg = GyroscopeConfig(api_key="test", llm=LLMConfig(max_concurrent=2))
    return LLMClient(cfg)


def _make_response(status: int) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


def _rate_limit_error() -> RateLimitError:
    return RateLimitError(
        "rate limited",
        response=_make_response(429),
        body=None,
    )


def _bad_request_error() -> BadRequestError:
    # Client-side mistake — must NOT be retried.
    return BadRequestError(
        "bad request",
        response=_make_response(400),
        body=None,
    )


def _connection_error() -> APIConnectionError:
    return APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]
        # Mimic the cache-usage fields so the DEBUG-logging branch is exercised.
        self.usage = type(
            "Usage",
            (),
            {"cache_read_input_tokens": 12, "cache_creation_input_tokens": 0},
        )()


# ---------------------------------------------------------------------------
# _is_retryable_llm_error — pure-function unit checks.
# ---------------------------------------------------------------------------


def test_is_retryable_llm_error_classifies_correctly() -> None:
    assert _is_retryable_llm_error(_rate_limit_error()) is True
    assert _is_retryable_llm_error(_connection_error()) is True
    assert _is_retryable_llm_error(httpx.ReadTimeout("slow")) is True

    # 4xx client errors must NOT be retried.
    assert _is_retryable_llm_error(_bad_request_error()) is False

    # Non-API errors are also not our problem.
    assert _is_retryable_llm_error(ValueError("oops")) is False


def test_is_retryable_llm_error_distinguishes_5xx_from_other_status_errors() -> None:
    five_oh_three = APIStatusError("service unavailable", response=_make_response(503), body=None)
    four_oh_four = APIStatusError("not found", response=_make_response(404), body=None)
    assert _is_retryable_llm_error(five_oh_three) is True
    assert _is_retryable_llm_error(four_oh_four) is False


# ---------------------------------------------------------------------------
# Retry behaviour wired through tenacity. We patch
# ``self._client.messages.create`` so no network is hit and we can count
# attempts directly. The tenacity wait is patched to zero so the test does
# not actually sleep through exponential backoff.
# ---------------------------------------------------------------------------


def _disable_backoff(client_obj: LLMClient) -> None:
    client_obj._complete_raw.retry.wait = lambda *a, **k: 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_complete_raw_retries_on_rate_limit_then_succeeds(client: LLMClient) -> None:
    attempts: list[int] = []

    async def _flaky(**_: Any) -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            raise _rate_limit_error()
        return _FakeResponse("ok")

    client._client.messages.create = AsyncMock(side_effect=_flaky)  # type: ignore[assignment]
    _disable_backoff(client)

    out = await client._complete_raw(
        model="x",
        system="s",
        messages=[{"role": "user", "content": "u"}],
        max_tokens=8,
        temperature=0.0,
    )
    assert out == "ok"
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_complete_raw_does_not_retry_on_400(client: LLMClient) -> None:
    attempts: list[int] = []

    async def _always_400(**_: Any) -> Any:
        attempts.append(1)
        raise _bad_request_error()

    client._client.messages.create = AsyncMock(side_effect=_always_400)  # type: ignore[assignment]
    _disable_backoff(client)

    with pytest.raises(BadRequestError):
        await client._complete_raw(
            model="x",
            system="s",
            messages=[{"role": "user", "content": "u"}],
            max_tokens=8,
            temperature=0.0,
        )
    # Exactly one attempt — no retry should have been issued for a 400.
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_complete_raw_retries_on_connection_error(client: LLMClient) -> None:
    attempts: list[int] = []

    async def _flaky(**_: Any) -> Any:
        attempts.append(1)
        if len(attempts) < 2:
            raise _connection_error()
        return _FakeResponse("done")

    client._client.messages.create = AsyncMock(side_effect=_flaky)  # type: ignore[assignment]
    _disable_backoff(client)

    out = await client._complete_raw(
        model="x",
        system="s",
        messages=[{"role": "user", "content": "u"}],
        max_tokens=8,
        temperature=0.0,
    )
    assert out == "done"
    assert len(attempts) == 2


# ---------------------------------------------------------------------------
# Role accessors.
# ---------------------------------------------------------------------------


def test_model_for_returns_role_specific_model(client: LLMClient) -> None:
    cfg = client.config.llm
    assert client.model_for("curator") == cfg.curator_model
    assert client.model_for("swarm") == cfg.swarm_model
    assert client.model_for("critic") == cfg.critic_model
    assert client.model_for("judge") == cfg.judge_model


def test_temperature_for_returns_role_specific_temperature(client: LLMClient) -> None:
    cfg = client.config.llm
    assert client.temperature_for("curator") == cfg.temperature_curator
    assert client.temperature_for("swarm") == cfg.temperature_swarm
    assert client.temperature_for("critic") == cfg.temperature_critic
    # Judge now has its own dedicated temperature knob.
    assert client.temperature_for("judge") == cfg.temperature_judge


def test_temperature_judge_is_independent_of_critic() -> None:
    """``temperature_judge`` must not piggy-back on ``temperature_critic``.

    Setting one must not affect the other — the two roles are tuned
    independently. Regression guard against the previous implementation
    which silently returned ``temperature_critic`` for the judge role.
    """
    cfg = GyroscopeConfig(
        api_key="test",
        llm=LLMConfig(temperature_critic=0.0, temperature_judge=0.42),
    )
    client_obj = LLMClient(cfg)
    assert client_obj.temperature_for("critic") == 0.0
    assert client_obj.temperature_for("judge") == pytest.approx(0.42)

    cfg2 = GyroscopeConfig(
        api_key="test",
        llm=LLMConfig(temperature_critic=0.3, temperature_judge=0.0),
    )
    client_obj2 = LLMClient(cfg2)
    assert client_obj2.temperature_for("critic") == pytest.approx(0.3)
    assert client_obj2.temperature_for("judge") == 0.0


def test_model_for_unknown_role_raises(client: LLMClient) -> None:
    with pytest.raises(ValueError, match="Unknown LLM role"):
        client.model_for("nonsense")


def test_temperature_for_unknown_role_raises(client: LLMClient) -> None:
    with pytest.raises(ValueError, match="Unknown LLM role"):
        client.temperature_for("nonsense")


def test_config_property_exposes_full_config(client: LLMClient) -> None:
    assert isinstance(client.config, GyroscopeConfig)
    # Property is read-only (no setter).
    with pytest.raises(AttributeError):
        client.config = GyroscopeConfig(api_key="x")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# complete_json strict=False behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_json_returns_default_when_not_strict(client: LLMClient) -> None:
    async def _garbage(**_: Any) -> Any:
        # The complete() wrapper prefills "{" then appends model output. We
        # return text that has no parsable JSON object so the parser raises.
        return _FakeResponse("not even close to json")

    client._client.messages.create = AsyncMock(side_effect=_garbage)  # type: ignore[assignment]

    sentinel = {"role": "fallback"}
    out = await client.complete_json(
        system="s",
        user="u",
        strict=False,
        default=sentinel,
    )
    assert out == sentinel


@pytest.mark.asyncio
async def test_complete_json_raises_when_strict(client: LLMClient) -> None:
    async def _garbage(**_: Any) -> Any:
        return _FakeResponse("not even close to json")

    client._client.messages.create = AsyncMock(side_effect=_garbage)  # type: ignore[assignment]

    with pytest.raises(ValueError):
        await client.complete_json(system="s", user="u")


# ---------------------------------------------------------------------------
# cache_system semantics.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_system_true_always_emits_cache_control_even_for_short(
    client: LLMClient,
) -> None:
    """``cache_system=True`` must mark the system block with ``cache_control``
    regardless of byte length. The previous 1000-char gate was the wrong
    unit (bytes, not tokens) and confused opt-in callers like LLMJudge.
    """
    captured: dict[str, Any] = {}

    async def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeResponse("ok")

    client._client.messages.create = AsyncMock(side_effect=_capture)  # type: ignore[assignment]

    short_system = "tiny system"  # well below the historical 1000-byte gate
    out = await client.complete(
        system=short_system,
        user="u",
        cache_system=True,
    )
    assert out == "ok"
    sys_arg = captured["system"]
    assert isinstance(sys_arg, list), "cache_system=True must emit a typed block list"
    assert sys_arg[0]["type"] == "text"
    assert sys_arg[0]["text"] == short_system
    assert sys_arg[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_cache_system_false_keeps_system_as_string(
    client: LLMClient,
) -> None:
    captured: dict[str, Any] = {}

    async def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeResponse("ok")

    client._client.messages.create = AsyncMock(side_effect=_capture)  # type: ignore[assignment]

    await client.complete(system="abcdef", user="u", cache_system=False)
    assert captured["system"] == "abcdef"


@pytest.mark.asyncio
async def test_cache_system_true_marks_cache_control_in_complete_messages(
    client: LLMClient,
) -> None:
    """The same semantics apply to :meth:`LLMClient.complete_messages`."""
    from gyroscope.core.llm import LLMMessage

    captured: dict[str, Any] = {}

    async def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeResponse("ok")

    client._client.messages.create = AsyncMock(side_effect=_capture)  # type: ignore[assignment]

    await client.complete_messages(
        system="short",
        messages=[LLMMessage(role="user", content="hi")],
        cache_system=True,
    )
    sys_arg = captured["system"]
    assert isinstance(sys_arg, list)
    assert sys_arg[0]["cache_control"] == {"type": "ephemeral"}
