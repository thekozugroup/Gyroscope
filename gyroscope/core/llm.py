"""Thin async wrapper around the Anthropic SDK.

Centralises:
- API key resolution
- async client lifecycle
- concurrency limiting (semaphore)
- retry with exponential backoff on rate limits / overload
- prompt caching of stable system prompts
- structured JSON helpers
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from anthropic import APIError, APIStatusError, AsyncAnthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gyroscope.core.config import GyroscopeConfig

logger = logging.getLogger(__name__)


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

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---------- core completion ----------

    @retry(
        retry=retry_if_exception_type((APIError, APIStatusError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
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
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
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
        """Single-turn completion. Optionally cache the system prompt."""
        cfg = self._config.llm
        model = model or cfg.swarm_model
        max_tokens = max_tokens or cfg.max_tokens
        temperature = cfg.temperature_swarm if temperature is None else temperature
        cache_system = cfg.cache_prompts if cache_system is None else cache_system

        sys_payload: str | list[dict[str, Any]]
        if cache_system and len(system) > 1000:
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
        cfg = self._config.llm
        model = model or cfg.swarm_model
        max_tokens = max_tokens or cfg.max_tokens
        temperature = cfg.temperature_swarm if temperature is None else temperature
        cache_system = cfg.cache_prompts if cache_system is None else cache_system

        sys_payload: str | list[dict[str, Any]]
        if cache_system and len(system) > 1000:
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
    ) -> Any:
        """Completion that must return a JSON object. We assistant-prefill `{`
        and then close-parse to tolerate trailing tokens."""
        raw = await self.complete(
            system=system,
            user=user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            cache_system=cache_system,
            assistant_prefill="{",
        )
        return _parse_json_object(raw)

    async def complete_json_array(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool | None = None,
    ) -> list[Any]:
        raw = await self.complete(
            system=system,
            user=user,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            cache_system=cache_system,
            assistant_prefill="[",
        )
        return _parse_json_array(raw)

    # ---------- convenience: bounded parallel map ----------

    async def gather(self, coros: list[Any]) -> list[Any]:
        """Run coroutines with the client's semaphore already bounding concurrency."""
        return await asyncio.gather(*coros)


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
