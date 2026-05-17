"""Persona generation tests using a fake LLM."""

from __future__ import annotations

import pytest

from gyroscope.sft.personas import generate_personas

from .conftest import FakeLLM, make_golden


@pytest.mark.asyncio
async def test_persona_ids_are_deterministic_and_padded():
    golden = make_golden()
    fake = FakeLLM(
        json_array_responses=[
            [
                {
                    "name": "Alex",
                    "description": "An eager novice.",
                    "expertise_level": "novice",
                    "tone": "eager and verbose",
                },
                {
                    "name": "Bea",
                    "description": "A seasoned pro.",
                    "expertise_level": "expert",
                    "tone": "curt and skeptical",
                },
                {
                    "name": "Cam",
                    "description": "Mid-level user.",
                    "expertise_level": "intermediate",
                    "tone": "polite",
                },
            ]
        ]
    )
    personas = await generate_personas(golden, fake, n=3)  # type: ignore[arg-type]

    assert [p.id for p in personas] == ["PER-0001", "PER-0002", "PER-0003"]
    assert personas[0].name == "Alex"
    assert personas[0].expertise_level == "novice"
    # at least one call made
    assert fake.json_array_calls, "expected the LLM array helper to be called"


@pytest.mark.asyncio
async def test_persona_count_padded_with_fallback_when_llm_returns_too_few():
    golden = make_golden()
    fake = FakeLLM(
        json_array_responses=[
            [
                {
                    "name": "Solo",
                    "description": "Only one persona returned.",
                    "expertise_level": "intermediate",
                    "tone": "neutral",
                }
            ]
        ]
    )
    personas = await generate_personas(golden, fake, n=4)  # type: ignore[arg-type]
    assert len(personas) == 4
    assert personas[0].name == "Solo"
    # All ids are still deterministic and unique.
    ids = [p.id for p in personas]
    assert ids == ["PER-0001", "PER-0002", "PER-0003", "PER-0004"]
    assert len(set(ids)) == len(ids)


@pytest.mark.asyncio
async def test_persona_invalid_expertise_coerced():
    golden = make_golden()
    fake = FakeLLM(
        json_array_responses=[
            [
                {
                    "name": "Weird",
                    "description": "Bad expertise tag.",
                    "expertise_level": "wizard",
                    "tone": "mystic",
                }
            ]
        ]
    )
    personas = await generate_personas(golden, fake, n=1)  # type: ignore[arg-type]
    assert personas[0].expertise_level == "intermediate"


@pytest.mark.asyncio
async def test_persona_zero_returns_empty():
    golden = make_golden()
    fake = FakeLLM()
    out = await generate_personas(golden, fake, n=0)  # type: ignore[arg-type]
    assert out == []
