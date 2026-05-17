"""Tests for the extractor's resilience and its use of the new public
``LLMClient`` accessors (``model_for`` / ``temperature_for``).

Covers:

* ``extract_identity`` falls back to the default Identity (rather than
  crashing the curation phase) when the LLM returns a payload that cannot
  be parsed as JSON.
* ``extract_principles`` reaches for the curator role via the new public
  accessors — never via the private ``_config`` attribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from gyroscope.core.config import GyroscopeConfig, LLMConfig
from gyroscope.core.models import Chunk
from gyroscope.curation.extractor import extract_identity, extract_principles


@dataclass
class _TrackingClient:
    """Stub LLM client that records every call to ``model_for`` and
    ``temperature_for``, plus the kwargs passed to ``complete_json`` /
    ``complete_json_array``.

    ``identity_raise`` controls what ``complete_json`` does:
      - None: returns ``identity_payload``;
      - an exception instance: raises it.
    """

    identity_payload: Any = None
    identity_raise: BaseException | None = None
    array_returns: list[list[dict[str, Any]]] = field(default_factory=list)

    model_for_calls: list[str] = field(default_factory=list)
    temperature_for_calls: list[str] = field(default_factory=list)
    array_calls: list[dict[str, Any]] = field(default_factory=list)
    object_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._config = GyroscopeConfig(
            api_key="test",
            llm=LLMConfig(curator_model="curator-x", temperature_curator=0.11),
        )
        self._iter = iter(self.array_returns)

    @property
    def config(self) -> GyroscopeConfig:
        return self._config

    def model_for(self, role: str) -> str:
        self.model_for_calls.append(role)
        return self._config.llm.curator_model  # extractor only asks for curator

    def temperature_for(self, role: str) -> float:
        self.temperature_for_calls.append(role)
        return self._config.llm.temperature_curator

    async def complete_json(self, **kwargs: Any) -> Any:
        self.object_calls.append(kwargs)
        if self.identity_raise is not None:
            raise self.identity_raise
        return self.identity_payload

    async def complete_json_array(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.array_calls.append(kwargs)
        try:
            return next(self._iter)
        except StopIteration:
            return []


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c-{i:04d}", document_source="src://x", text=f"text {i}", order=i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_identity_falls_back_on_parse_error() -> None:
    """A malformed model output must NOT crash the curation phase — instead
    the extractor logs and returns the default Identity."""
    client = _TrackingClient(
        identity_raise=ValueError("No JSON object found in response: '???'"),
    )
    identity = await extract_identity(_chunks(3), client)
    assert identity.role == "Unspecified Role"
    assert identity.mission == "No mission defined."
    # We DID attempt the LLM call (so the fallback was triggered defensively,
    # not by the empty-chunks short-circuit).
    assert len(client.object_calls) == 1


@pytest.mark.asyncio
async def test_extract_identity_falls_back_when_payload_is_not_object() -> None:
    """If the LLM returns a JSON array (or scalar) where an object was
    required, the extractor logs a warning and falls back instead of
    propagating a TypeError up through the pipeline."""
    client = _TrackingClient(identity_payload=["wrong shape"])
    identity = await extract_identity(_chunks(3), client)
    assert identity.role == "Unspecified Role"


# ---------------------------------------------------------------------------
# Use of the new public role accessors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extractor_uses_model_for_and_temperature_for() -> None:
    """The extractor must look up the curator model/temperature via the new
    public ``model_for`` / ``temperature_for`` accessors — not via
    ``client._config.llm``."""
    chunks = _chunks(3)
    client = _TrackingClient(
        array_returns=[
            [
                {
                    "statement": "Always verify quantities before certifying.",
                    "source_chunk_ids": ["c-0000"],
                }
            ]
        ],
    )
    await extract_principles(chunks, client, batch_size=10)

    # Exactly one curator lookup for model and one for temperature per batch.
    assert client.model_for_calls == ["curator"]
    assert client.temperature_for_calls == ["curator"]

    # And those values were forwarded into the LLM call kwargs verbatim.
    assert len(client.array_calls) == 1
    call = client.array_calls[0]
    assert call["model"] == "curator-x"
    assert call["temperature"] == pytest.approx(0.11)
