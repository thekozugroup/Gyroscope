"""Scenario generation must fan out in parallel.

Sequential ``await`` left the LLM client semaphore (sized by
``LLMConfig.max_concurrent``) idle for almost the entire generation phase.
For a 5000-trajectory run that's the difference between a few minutes and
several hours of wall-clock just for scenarios.

This test forces multiple scenario coroutines to make their LLM call before
ANY of them resolves — only possible if the caller is using ``gather``
(or equivalent fan-out) rather than a serial for-loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gyroscope.core.models import Persona
from gyroscope.sft.scenarios import generate_scenarios

from .conftest import make_golden


def _personas(n: int = 2) -> list[Persona]:
    return [
        Persona(
            id=f"PER-{i:04d}",
            name=f"P{i}",
            description=f"persona {i}",
            expertise_level="intermediate",
            tone="neutral",
        )
        for i in range(1, n + 1)
    ]


class ConcurrencyTrackingLLM:
    """Stub LLM that records peak in-flight ``complete_json`` calls.

    Each call increments an in-flight counter, awaits a fresh event, and
    decrements. If the test reaches the expected in-flight count, it sets
    the event so every coroutine completes; otherwise it times out.
    """

    def __init__(self, expected_inflight: int, n_total: int) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.call_count = 0
        self._expected = expected_inflight
        self._n_total = n_total
        self._release = asyncio.Event()

    async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        if self.in_flight >= self._expected:
            # Enough coroutines are parked in-flight together — proof of
            # parallel fan-out. Let everyone finish.
            self._release.set()
        try:
            # Wait until the test has observed the parallel fan-out, then
            # return a canned response.
            await asyncio.wait_for(self._release.wait(), timeout=2.0)
        finally:
            self.in_flight -= 1
        return {"prompt_seed": f"seed-{self.call_count}"}


@pytest.mark.asyncio
async def test_generate_scenarios_runs_in_parallel():
    golden = make_golden(n_procedures=2, n_principles=2)
    personas = _personas(2)
    n_total = 4

    # Expect at least 2 coroutines to be in-flight at once (proof of fan-out).
    fake = ConcurrencyTrackingLLM(expected_inflight=2, n_total=n_total)

    scenarios = await generate_scenarios(
        golden, personas, fake, n_total, {"medium": 1.0}  # type: ignore[arg-type]
    )

    assert len(scenarios) == n_total
    # The sequential implementation would never have >1 in-flight call.
    assert fake.peak_in_flight >= 2, (
        f"expected parallel fan-out; peak in-flight was {fake.peak_in_flight}"
    )
    # Order must still be deterministic by scenario index.
    assert [s.id for s in scenarios] == [
        f"SCN-{i:04d}" for i in range(1, n_total + 1)
    ]


@pytest.mark.asyncio
async def test_generate_scenarios_creates_all_coros_before_first_finishes():
    """A stricter version: every coroutine should have started its LLM call
    before any of them is allowed to finish."""
    golden = make_golden(n_procedures=2, n_principles=2)
    personas = _personas(2)
    n_total = 5

    fake = ConcurrencyTrackingLLM(expected_inflight=n_total, n_total=n_total)

    scenarios = await generate_scenarios(
        golden, personas, fake, n_total, {"medium": 1.0}  # type: ignore[arg-type]
    )

    assert len(scenarios) == n_total
    assert fake.peak_in_flight == n_total, (
        f"all {n_total} coros must be in-flight together; peak was "
        f"{fake.peak_in_flight}"
    )
