"""Tests that ``_run_batched_array`` issues its per-batch LLM calls
concurrently rather than serialising them.

Each extractor batches its chunks and used to ``await`` one batch at a
time, leaving the rest of the ``LLMClient`` semaphore idle. The fix
dispatches batches via ``asyncio.gather`` so a single extractor can have
several batches in flight at once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from gyroscope.core.config import GyroscopeConfig, LLMConfig
from gyroscope.core.models import Chunk
from gyroscope.curation.extractor import _run_batched_array, extract_principles


@dataclass
class _ConcurrencyTrackingLLM:
    """Stub LLM that records concurrent ``complete_json_array`` calls.

    ``in_flight`` is incremented on call entry and decremented on exit.
    ``peak_in_flight`` records the maximum observed concurrency, which the
    tests assert against to prove batches run in parallel.
    """

    delay: float = 0.05
    return_payload: list[dict[str, Any]] = field(default_factory=list)

    in_flight: int = 0
    peak_in_flight: int = 0
    call_count: int = 0
    array_calls: list[dict[str, Any]] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self._config = GyroscopeConfig(api_key="test", llm=LLMConfig())

    def model_for(self, role: str) -> str:
        return self._config.llm.curator_model

    def temperature_for(self, role: str) -> float:
        return self._config.llm.temperature_curator

    async def complete_json_array(self, **kwargs: Any) -> list[dict[str, Any]]:
        async with self._lock:
            self.in_flight += 1
            self.call_count += 1
            if self.in_flight > self.peak_in_flight:
                self.peak_in_flight = self.in_flight
            self.array_calls.append(kwargs)
        try:
            # Yield control long enough that other batches can enter before
            # this one finishes — without this, a fast-enough completion
            # could let the scheduler run them serially even under gather.
            await asyncio.sleep(self.delay)
        finally:
            async with self._lock:
                self.in_flight -= 1
        return list(self.return_payload)


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c-{i:04d}", document_source="src://x", text=f"text {i}", order=i)
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_run_batched_array_dispatches_batches_concurrently() -> None:
    """With >=3 batches in a single extractor call, at least 2 must be
    in flight at the same instant."""
    chunks = _chunks(9)  # 3 batches at batch_size=3
    client = _ConcurrencyTrackingLLM()

    await _run_batched_array(
        chunks=chunks,
        client=client,
        system_prompt="sys",
        batch_size=3,
        instruction="instr",
    )

    assert client.call_count == 3, "expected one LLM call per batch"
    assert client.peak_in_flight >= 2, (
        f"batches ran serially (peak_in_flight={client.peak_in_flight}); "
        "expected concurrent dispatch"
    )


@pytest.mark.asyncio
async def test_run_batched_array_preserves_input_order() -> None:
    """``asyncio.gather`` returns results in input order; the flattened
    output must therefore mirror the order of the input chunks."""
    chunks = _chunks(6)  # 3 batches at batch_size=2

    @dataclass
    class _OrderedClient:
        in_flight: int = 0
        peak_in_flight: int = 0

        def __post_init__(self) -> None:
            self._config = GyroscopeConfig(api_key="test", llm=LLMConfig())

        def model_for(self, role: str) -> str:
            return self._config.llm.curator_model

        def temperature_for(self, role: str) -> float:
            return self._config.llm.temperature_curator

        async def complete_json_array(self, **kwargs: Any) -> list[dict[str, Any]]:
            # The user prompt contains the chunk ids in this batch; tag the
            # first chunk id back into the payload so we can verify order
            # without relying on internal call ordering.
            user_text: str = kwargs["user"]
            first_id_marker = "c-"
            idx = user_text.find(first_id_marker)
            tag = user_text[idx : idx + len(first_id_marker) + 4]
            return [{"tag": tag}]

    client = _OrderedClient()
    out = await _run_batched_array(
        chunks=chunks,
        client=client,
        system_prompt="sys",
        batch_size=2,
        instruction="instr",
    )

    assert [item["tag"] for item in out] == ["c-0000", "c-0002", "c-0004"]


@pytest.mark.asyncio
async def test_extract_principles_runs_batches_concurrently() -> None:
    """The same concurrency property must hold end-to-end through the
    public ``extract_principles`` entry point."""
    chunks = _chunks(9)
    client = _ConcurrencyTrackingLLM()
    await extract_principles(chunks, client, batch_size=3)
    assert client.call_count == 3
    assert client.peak_in_flight >= 2
