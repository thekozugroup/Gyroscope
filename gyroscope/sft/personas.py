"""Persona generation seeded from the GoldenDocument role identity."""

from __future__ import annotations

import logging
from typing import Any, Literal, get_args

from gyroscope.core.llm import LLMClient
from gyroscope.core.models import GoldenDocument, Persona

logger = logging.getLogger(__name__)


_ExpertiseLevel = Literal["novice", "intermediate", "expert"]


_PERSONA_SYSTEM = """You design realistic *user* personas who would interact with a domain-expert AI agent.

Given the agent's identity, mission, and a sample of its principles, return a JSON array of
diverse personas. Span expertise levels and tones. Each persona must be plausible for
someone who would actually talk to this agent.

Return JSON only. Schema:
[
  {
    "name": "short label",
    "description": "1-2 sentences describing who they are and what they typically want",
    "expertise_level": "novice" | "intermediate" | "expert",
    "tone": "short adjective phrase such as 'curt and skeptical' or 'eager and verbose'"
  }
]
""".strip()


def _persona_id(index: int) -> str:
    return f"PER-{index:04d}"


def _build_user_prompt(golden: GoldenDocument, n: int) -> str:
    principles_sample = "\n".join(f"- [{p.id}] {p.statement}" for p in golden.principles[:8])
    procedures_sample = "\n".join(
        f"- [{proc.id}] {proc.name}: {proc.purpose}" for proc in golden.procedures[:6]
    )
    return (
        f"AGENT ROLE: {golden.identity.role}\n"
        f"AGENT DESCRIPTION: {golden.identity.description}\n"
        f"MISSION: {golden.identity.mission}\n\n"
        f"PRINCIPLES (sample):\n{principles_sample or '(none)'}\n\n"
        f"PROCEDURES (sample):\n{procedures_sample or '(none)'}\n\n"
        f"Generate EXACTLY {n} distinct personas as a JSON array. "
        f"Spread expertise levels (novice/intermediate/expert) roughly evenly. "
        f"Tones should vary."
    )


def _coerce_expertise(value: Any) -> _ExpertiseLevel:
    allowed = get_args(_ExpertiseLevel)
    if isinstance(value, str) and value in allowed:
        return value  # type: ignore[return-value]
    return "intermediate"


def _personas_from_raw(raw: list[Any], n: int) -> list[Persona]:
    personas: list[Persona] = []
    for idx, item in enumerate(raw[:n], start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"Persona {idx}")).strip() or f"Persona {idx}"
        description = str(item.get("description", "")).strip() or name
        expertise = _coerce_expertise(item.get("expertise_level"))
        tone = str(item.get("tone", "neutral")).strip() or "neutral"
        personas.append(
            Persona(
                id=_persona_id(idx),
                name=name,
                description=description,
                expertise_level=expertise,
                tone=tone,
            )
        )
    return personas


def _fallback_personas(n: int, golden: GoldenDocument) -> list[Persona]:
    """Deterministic backstop if the LLM ever returns nothing usable."""
    role = golden.identity.role
    expertises: list[_ExpertiseLevel] = ["novice", "intermediate", "expert"]
    tones = [
        "curious and verbose",
        "curt and skeptical",
        "polite and methodical",
        "anxious and hurried",
        "analytical and precise",
    ]
    personas: list[Persona] = []
    for idx in range(1, n + 1):
        exp = expertises[(idx - 1) % len(expertises)]
        tone = tones[(idx - 1) % len(tones)]
        personas.append(
            Persona(
                id=_persona_id(idx),
                name=f"{exp.capitalize()} stakeholder {idx}",
                description=(f"A {exp} who interacts with a {role}. Tone: {tone}."),
                expertise_level=exp,
                tone=tone,
            )
        )
    return personas


async def generate_personas(
    golden: GoldenDocument,
    client: LLMClient,
    n: int,
) -> list[Persona]:
    """Generate `n` distinct personas seeded from the golden document's identity."""
    if n <= 0:
        return []

    user_prompt = _build_user_prompt(golden, n)
    try:
        raw = await client.complete_json_array(
            system=_PERSONA_SYSTEM,
            user=user_prompt,
            cache_system=True,
            temperature=0.7,
        )
    except Exception as exc:
        logger.warning("Persona LLM call failed (%s); using deterministic fallback.", exc)
        return _fallback_personas(n, golden)

    personas = _personas_from_raw(raw, n)
    if len(personas) < n:
        # Top up with deterministic fallbacks to honour the requested count.
        extras = _fallback_personas(n, golden)
        existing = {p.id for p in personas}
        for extra in extras:
            if extra.id in existing:
                continue
            personas.append(extra)
            if len(personas) >= n:
                break
    return personas[:n]
