"""Thin async wrapper around the Anthropic SDK.

Centralises:
- API key resolution
- async client lifecycle
- concurrency limiting (semaphore)
- retry with exponential backoff on transient failures only
- prompt caching of stable system prompts
- structured JSON helpers
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx
from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    RateLimitError,
)
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
)

from gyroscope.core.config import GyroscopeConfig
from gyroscope.core.retry import BACKOFF_BASE, BACKOFF_CAP, MAX_ATTEMPTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry discriminator
# ---------------------------------------------------------------------------


# HTTP status codes that represent transient server-side conditions worth
# retrying. 4xx codes that indicate client error (400 / 401 / 403 / 404 / 422)
# are intentionally absent — retrying them just wastes tokens.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Optional: APITimeoutError may not exist on older SDK versions.
_APITimeoutError: type[BaseException] | None = getattr(anthropic, "APITimeoutError", None)


# Per-instance state for decorrelated jitter. Keyed by the RetryCallState
# object identity (tenacity creates a fresh state per decorated call) so
# concurrent retries on different requests do not interfere.
_PREV_WAIT_BY_STATE: dict[int, float] = {}


def _decorrelated_jitter_wait(state: RetryCallState) -> float:
    """Decorrelated-jitter backoff.

    AWS-style decorrelated jitter: ``delay_n = min(cap, U(base, prev_wait * 3))``
    where ``prev_wait`` is the wait FROM THE LAST ATTEMPT (not the cumulative
    idle time across all attempts — that's what tenacity's ``state.idle_for``
    holds, and using it would compound the spread on attempts ≥ 3 faster
    than the formula intends).

    Spreads retry timings aggressively so a synchronised 429 burst across
    many concurrent workers does not all retry at the same wall-clock tick.
    """
    key = id(state)
    if state.attempt_number <= 1:
        _PREV_WAIT_BY_STATE.pop(key, None)
        next_wait = BACKOFF_BASE
    else:
        prev_wait = _PREV_WAIT_BY_STATE.get(key, BACKOFF_BASE)
        next_wait = min(BACKOFF_CAP, random.uniform(BACKOFF_BASE, prev_wait * 3.0))
    _PREV_WAIT_BY_STATE[key] = next_wait
    # Bound the state map so a long-running process doesn't accumulate
    # entries from many short retry chains. Eight is fine — typical depth
    # is 1-4 entries since states are released after the retry resolves.
    if len(_PREV_WAIT_BY_STATE) > 64:
        # Drop the oldest insertion (py3.7+ preserves order).
        oldest = next(iter(_PREV_WAIT_BY_STATE))
        if oldest != key:
            _PREV_WAIT_BY_STATE.pop(oldest, None)
    return next_wait


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` is a transient API failure worth retrying.

    We deliberately do NOT retry generic :class:`anthropic.APIError` or
    4xx client errors — those indicate a request the caller must fix.
    """
    if isinstance(exc, RateLimitError | APIConnectionError):
        return True
    if _APITimeoutError is not None and isinstance(exc, _APITimeoutError):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in _RETRYABLE_STATUS_CODES
    return isinstance(exc, httpx.TransportError)


# ---------------------------------------------------------------------------
# Role -> model / temperature mapping
# ---------------------------------------------------------------------------


_VALID_ROLES: frozenset[str] = frozenset({"curator", "swarm", "critic", "judge"})


@dataclass
class LLMMessage:
    role: str  # "user" | "assistant"
    content: str


class LLMClient:
    """Async Anthropic wrapper with caching, retries and bounded concurrency."""

    def __init__(self, config: GyroscopeConfig) -> None:
        self._config = config
        self._client = AsyncAnthropic(api_key=config.resolved_api_key())
        self._sem = asyncio.Semaphore(config.llm.max_concurrent)

    # ---------- public config access ----------

    @property
    def config(self) -> GyroscopeConfig:
        """Read-only access to the underlying :class:`GyroscopeConfig`."""
        return self._config

    def model_for(self, role: str) -> str:
        """Return the configured model id for ``role``.

        Roles: ``"curator" | "swarm" | "critic" | "judge"``. Unknown roles
        raise :class:`ValueError`.
        """
        cfg = self._config.llm
        if role == "curator":
            return cfg.curator_model
        if role == "swarm":
            return cfg.swarm_model
        if role == "critic":
            return cfg.critic_model
        if role == "judge":
            return cfg.judge_model
        raise ValueError(f"Unknown LLM role {role!r}; expected one of {sorted(_VALID_ROLES)}.")

    def temperature_for(self, role: str) -> float:
        """Return the configured sampling temperature for ``role``.

        The judge has its own ``temperature_judge`` knob (defaulting to 0.0
        for deterministic judging) so it can be tuned independently of the
        critic. Unknown roles raise :class:`ValueError`.
        """
        cfg = self._config.llm
        if role == "curator":
            return cfg.temperature_curator
        if role == "swarm":
            return cfg.temperature_swarm
        if role == "critic":
            return cfg.temperature_critic
        if role == "judge":
            return cfg.temperature_judge
        raise ValueError(f"Unknown LLM role {role!r}; expected one of {sorted(_VALID_ROLES)}.")

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---------- core completion ----------

    @retry(
        retry=retry_if_exception(_is_retryable_llm_error),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=_decorrelated_jitter_wait,
        reraise=True,
    )
    async def _complete_raw(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        async with self._sem:
            resp = await self._client.messages.create(
                model=model,
                system=system,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
            )
        _log_cache_usage(resp)
        parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool | None = None,
        assistant_prefill: str | None = None,
    ) -> str:
        """Single-turn completion. Optionally cache the system prompt.

        When ``cache_system`` is True (default from ``LLMConfig.cache_prompts``)
        we always emit the ``cache_control={"type": "ephemeral"}`` marker on
        the system block. Anthropic only actually caches blocks above the
        provider's minimum (currently ~1024 tokens) and silently returns a
        cache miss for anything smaller — callers are responsible for ensuring
        the block is large enough to be worth caching. This is closer to the
        SDK contract than the previous byte-length heuristic.
        """
        cfg = self._config.llm
        model = model or cfg.swarm_model
        max_tokens = max_tokens or cfg.max_tokens
        temperature = cfg.temperature_swarm if temperature is None else temperature
        cache_system = cfg.cache_prompts if cache_system is None else cache_system

        sys_payload: str | list[dict[str, Any]]
        if cache_system:
            sys_payload = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            sys_payload = system

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})

        out = await self._complete_raw(
            model=model,
            system=sys_payload,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if assistant_prefill:
            return assistant_prefill + out
        return out

    async def complete_messages(
        self,
        *,
        system: str,
        messages: list[LLMMessage],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool | None = None,
    ) -> str:
        """Multi-turn completion. See :meth:`complete` for ``cache_system``
        semantics — when True we always emit the ``cache_control`` marker and
        leave it to Anthropic (and the caller) to decide whether the block is
        large enough to actually cache.
        """
        cfg = self._config.llm
        model = model or cfg.swarm_model
        max_tokens = max_tokens or cfg.max_tokens
        temperature = cfg.temperature_swarm if temperature is None else temperature
        cache_system = cfg.cache_prompts if cache_system is None else cache_system

        sys_payload: str | list[dict[str, Any]]
        if cache_system:
            sys_payload = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            sys_payload = system

        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        return await self._complete_raw(
            model=model,
            system=sys_payload,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ---------- JSON helpers ----------

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool | None = None,
        strict: bool = True,
        default: Any = None,
    ) -> Any:
        """Completion that must return a JSON object.

        We assistant-prefill ``{`` and then close-parse to tolerate trailing
        tokens. If ``strict`` is False, return ``default`` instead of raising
        on a parse failure.
        """
        raw = await self.complete(
            system=system,
            user=user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            cache_system=cache_system,
            assistant_prefill="{",
        )
        try:
            return _parse_json_object(raw)
        except (ValueError, json.JSONDecodeError):
            if strict:
                raise
            logger.warning("complete_json failed to parse model output; returning default.")
            return default

    async def complete_json_array(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool | None = None,
        strict: bool = True,
        default: list[Any] | None = None,
    ) -> list[Any]:
        """Completion that must return a JSON array.

        If ``strict`` is False, return ``default`` (or ``[]`` when ``default``
        is None) instead of raising on a parse failure.
        """
        raw = await self.complete(
            system=system,
            user=user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            cache_system=cache_system,
            assistant_prefill="[",
        )
        try:
            return _parse_json_array(raw)
        except (ValueError, json.JSONDecodeError):
            if strict:
                raise
            logger.warning("complete_json_array failed to parse model output; returning default.")
            return default if default is not None else []


# ---------------------------------------------------------------------------
# Prompt-cache observability — fully best-effort, never raises.
# ---------------------------------------------------------------------------


def _log_cache_usage(resp: Any) -> None:
    """Emit a DEBUG line with prompt-cache hit/miss counters, if available.

    Wrapped in a broad try/except — observability must never crash a call.
    """
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        cache_creation = getattr(usage, "cache_creation_input_tokens", None)
        if cache_read is None and cache_creation is None:
            return
        logger.debug(
            "anthropic prompt cache usage: read=%s creation=%s",
            cache_read,
            cache_creation,
        )
    except Exception:
        # Observability must never fail the call.
        logger.debug("Failed to read prompt-cache usage fields.", exc_info=True)


# ---------------------------------------------------------------------------
# JSON parsing helpers — tolerant to model chatter around the object.
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fence(s: str) -> str:
    m = _FENCE_RE.search(s)
    return m.group(1) if m else s


def _parse_json_object(s: str) -> Any:
    s = _strip_fence(s).strip()
    if not s.startswith("{"):
        idx = s.find("{")
        if idx == -1:
            raise ValueError(f"No JSON object found in response: {s[:200]!r}")
        s = s[idx:]
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(s)
    return obj


def _parse_json_array(s: str) -> list[Any]:
    s = _strip_fence(s).strip()
    if not s.startswith("["):
        idx = s.find("[")
        if idx == -1:
            raise ValueError(f"No JSON array found in response: {s[:200]!r}")
        s = s[idx:]
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(s)
    if not isinstance(obj, list):
        raise ValueError("Expected JSON array.")
    return obj
