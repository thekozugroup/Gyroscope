"""Scenario generation with stratified procedure x persona x difficulty sampling."""

from __future__ import annotations

import logging
import random
from typing import Any, Literal, get_args

from gyroscope.core.llm import LLMClient
from gyroscope.core.models import GoldenDocument, Persona, Procedure, Scenario

logger = logging.getLogger(__name__)


_Difficulty = Literal["easy", "medium", "hard", "adversarial"]


_SCENARIO_SYSTEM = """You design realistic *opening user prompts* (`prompt_seed`) for trajectories
that will train an AI agent in a specific role.

Each opening prompt is the kernel of one conversation. It must:
- sound like something the named persona would actually say,
- be answerable using the agent's principles/procedures,
- match the requested difficulty,
- be self-contained (no references to "the document" or "your training").

Return a single JSON object. Schema:
{ "prompt_seed": "the user's opening message, 1-4 sentences" }
""".strip()


def _scenario_id(index: int) -> str:
    return f"SCN-{index:04d}"


def _difficulty_targets(
    n_total: int, difficulty_mix: dict[str, float]
) -> dict[_Difficulty, int]:
    """Convert a difficulty mix into integer counts that sum to `n_total`."""
    allowed = get_args(_Difficulty)
    weights = {d: max(0.0, float(difficulty_mix.get(d, 0.0))) for d in allowed}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        # default: medium-only
        return {d: (n_total if d == "medium" else 0) for d in allowed}  # type: ignore[misc]

    raw_counts = {d: weights[d] / total_weight * n_total for d in allowed}
    floors = {d: int(raw_counts[d]) for d in allowed}
    remainder = n_total - sum(floors.values())
    # Distribute remainder by largest fractional part, deterministic tie-break by order.
    fractions = sorted(
        ((raw_counts[d] - floors[d], idx, d) for idx, d in enumerate(allowed)),
        key=lambda t: (-t[0], t[1]),
    )
    counts: dict[_Difficulty, int] = dict(floors)  # type: ignore[assignment]
    for i in range(remainder):
        _, _, d = fractions[i % len(fractions)]
        counts[d] += 1  # type: ignore[index]
    return counts


def _build_anchors(golden: GoldenDocument) -> list[tuple[str | None, list[str]]]:
    """Procedure-first, principle-fallback anchors.

    Returns list of (procedure_id_or_None, principle_ids).
    Procedures are used in declaration order; if there are no procedures, each
    principle becomes its own anchor.
    """
    if golden.procedures:
        anchors: list[tuple[str | None, list[str]]] = []
        for proc in golden.procedures:
            # Attach up to 3 principles as suggested ones (best-effort thematic linkage).
            suggested = [p.id for p in golden.principles[:3]]
            anchors.append((proc.id, suggested))
        return anchors
    if golden.principles:
        return [(None, [p.id]) for p in golden.principles]
    # No anchors at all — emit a single null anchor so we still produce scenarios.
    return [(None, [])]


def _round_robin_anchors(
    anchors: list[tuple[str | None, list[str]]], n_total: int
) -> list[tuple[str | None, list[str]]]:
    """Cycle anchors so all procedures get coverage at least once before repeats."""
    if not anchors:
        return []
    out: list[tuple[str | None, list[str]]] = []
    for i in range(n_total):
        out.append(anchors[i % len(anchors)])
    return out


def _build_user_prompt(
    golden: GoldenDocument,
    persona: Persona,
    difficulty: _Difficulty,
    procedure: Procedure | None,
    principle_ids: list[str],
) -> str:
    if procedure is not None:
        anchor_block = (
            f"PROCEDURE [{procedure.id}] {procedure.name}\n"
            f"PURPOSE: {procedure.purpose}\n"
        )
    elif principle_ids:
        anchor_block = "PRINCIPLES: " + ", ".join(principle_ids)
    else:
        anchor_block = "(no specific anchor)"

    difficulty_hint = {
        "easy": "a simple, common question",
        "medium": "a realistic everyday request requiring a thoughtful answer",
        "hard": "a tricky edge case requiring careful reasoning",
        "adversarial": "a probing or borderline request that tests the agent's principles",
    }[difficulty]

    return (
        f"AGENT ROLE: {golden.identity.role}\n"
        f"AGENT MISSION: {golden.identity.mission}\n\n"
        f"PERSONA: {persona.name} ({persona.expertise_level}, tone: {persona.tone})\n"
        f"PERSONA DETAIL: {persona.description}\n\n"
        f"ANCHOR:\n{anchor_block}\n\n"
        f"DIFFICULTY: {difficulty} — {difficulty_hint}.\n\n"
        f"Return JSON {{\"prompt_seed\": \"...\"}} only."
    )


def _fallback_prompt_seed(
    persona: Persona,
    difficulty: _Difficulty,
    procedure: Procedure | None,
    golden: GoldenDocument,
) -> str:
    target = procedure.name if procedure else golden.identity.role
    return (
        f"As a {persona.expertise_level} ({persona.tone}), I have a {difficulty} "
        f"question about {target}: could you walk me through how this works?"
    )


async def _generate_one_scenario(
    *,
    index: int,
    golden: GoldenDocument,
    persona: Persona,
    difficulty: _Difficulty,
    procedure: Procedure | None,
    principle_ids: list[str],
    client: LLMClient,
) -> Scenario:
    user_prompt = _build_user_prompt(
        golden, persona, difficulty, procedure, principle_ids
    )
    prompt_seed: str
    try:
        obj = await client.complete_json(
            system=_SCENARIO_SYSTEM,
            user=user_prompt,
            cache_system=True,
            temperature=0.8,
        )
        prompt_seed = str(obj.get("prompt_seed", "")).strip()
    except Exception as exc:
        logger.warning("Scenario LLM call failed (%s); using fallback seed.", exc)
        prompt_seed = ""

    if not prompt_seed:
        prompt_seed = _fallback_prompt_seed(persona, difficulty, procedure, golden)

    return Scenario(
        id=_scenario_id(index),
        procedure_id=procedure.id if procedure else None,
        principle_ids=list(principle_ids),
        persona_id=persona.id,
        difficulty=difficulty,
        prompt_seed=prompt_seed,
    )


async def generate_scenarios(
    golden: GoldenDocument,
    personas: list[Persona],
    client: LLMClient,
    n_total: int,
    difficulty_mix: dict[str, float],
) -> list[Scenario]:
    """Generate `n_total` scenarios stratified by difficulty and round-robin over anchors.

    Round-robin ensures every procedure (or principle when there are no procedures)
    gets at least one scenario before any anchor is reused.
    """
    if n_total <= 0 or not personas:
        return []

    targets = _difficulty_targets(n_total, difficulty_mix)
    # Flatten into an ordered list of difficulties such that we interleave them,
    # but produce *exactly* the requested counts.
    difficulty_pool: list[_Difficulty] = []
    for d in get_args(_Difficulty):
        difficulty_pool.extend([d] * targets.get(d, 0))  # type: ignore[arg-type]
    # Stable interleave: sort using a deterministic pseudo-shuffle.
    rng = random.Random(13)
    rng.shuffle(difficulty_pool)

    anchors = _round_robin_anchors(_build_anchors(golden), n_total)
    proc_by_id = {p.id: p for p in golden.procedures}

    scenarios: list[Scenario] = []
    for i in range(n_total):
        difficulty = difficulty_pool[i] if i < len(difficulty_pool) else "medium"
        anchor_pid, principle_ids = anchors[i]
        procedure = proc_by_id.get(anchor_pid) if anchor_pid else None
        persona = personas[i % len(personas)]
        scenarios.append(
            await _generate_one_scenario(
                index=i + 1,
                golden=golden,
                persona=persona,
                difficulty=difficulty,
                procedure=procedure,
                principle_ids=principle_ids,
                client=client,
            )
        )
    return scenarios


# Re-exported for tests / pipeline visibility.
def difficulty_targets(
    n_total: int, difficulty_mix: dict[str, float]
) -> dict[str, int]:
    """Public wrapper for the internal stratification helper."""
    return dict(_difficulty_targets(n_total, difficulty_mix))


def round_robin_anchors(
    anchors: list[tuple[str | None, list[str]]], n_total: int
) -> list[tuple[str | None, list[str]]]:
    """Public wrapper for the round-robin helper."""
    return _round_robin_anchors(anchors, n_total)


def build_anchors(golden: GoldenDocument) -> list[tuple[str | None, list[str]]]:
    """Public wrapper exposing anchor construction (procedure-first)."""
    return _build_anchors(golden)


__all__ = [
    "build_anchors",
    "difficulty_targets",
    "generate_scenarios",
    "round_robin_anchors",
]


# Type-checker hint that we use Any in fallback wrapper signatures.
_ = Any
