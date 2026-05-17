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
    five_oh_three = APIStatusError(
        "service unavailable", response=_make_response(503), body=None
    )
    four_oh_four = APIStatusError(
        "not found", response=_make_response(404), body=None
    )
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
    # Judge reuses critic temperature (deterministic judging).
    assert client.temperature_for("judge") == cfg.temperature_critic


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
