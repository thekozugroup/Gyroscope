"""Short-lived 4-agent rollout that produces a single Trajectory.

Agents:
- Planner   — picks an outline and target principle/procedure ids.
- User-sim  — plays the user (persona-consistent, multi-turn).
- Assistant — plays the role using the golden doc as context.
- Critic    — scores principle adherence, no hallucination, format.

A single repair pass (re-roll the last assistant turn) is attempted when the
critic score falls below `SFTConfig.critic_min_score`, up to
`SFTConfig.max_repair_attempts` times.
"""

from __future__ import annotations

import logging
from typing import Any

from gyroscope.core.config import SFTConfig
from gyroscope.core.llm import LLMClient, LLMMessage
from gyroscope.core.models import (
    GoldenDocument,
    Persona,
    Principle,
    Procedure,
    Scenario,
    Trajectory,
    TrajectoryMessage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt construction for the *assistant* (the role being trained).
# ---------------------------------------------------------------------------


def _select_principles(
    golden: GoldenDocument, scenario: Scenario, max_n: int = 8
) -> list[Principle]:
    """Pick principles the assistant must demonstrate for this scenario."""
    by_id = {p.id: p for p in golden.principles}
    chosen: list[Principle] = []
    for pid in scenario.principle_ids:
        if pid in by_id and by_id[pid] not in chosen:
            chosen.append(by_id[pid])
    # Top up with the first principles in the doc so the prompt is always useful.
    for p in golden.principles:
        if p in chosen:
            continue
        chosen.append(p)
        if len(chosen) >= max_n:
            break
    return chosen[:max_n]


def _selected_procedure(
    golden: GoldenDocument, scenario: Scenario
) -> Procedure | None:
    if not scenario.procedure_id:
        return None
    for p in golden.procedures:
        if p.id == scenario.procedure_id:
            return p
    return None


def build_system_prompt(golden: GoldenDocument, scenario: Scenario) -> str:
    """Construct the assistant's system prompt for a scenario."""
    parts: list[str] = []
    parts.append(f"# Identity\nYou are: {golden.identity.role}.")
    parts.append(golden.identity.description)
    parts.append(f"\n# Mission\n{golden.identity.mission}")

    principles = _select_principles(golden, scenario)
    if principles:
        parts.append("\n# Principles you MUST follow")
        for p in principles:
            parts.append(f"- [{p.id}] {p.statement}")

    procedure = _selected_procedure(golden, scenario)
    if procedure is not None:
        parts.append(f"\n# Relevant procedure: [{procedure.id}] {procedure.name}")
        parts.append(f"Purpose: {procedure.purpose}")
        if procedure.preconditions:
            parts.append("Preconditions:")
            for pc in procedure.preconditions:
                parts.append(f"- {pc}")
        parts.append("Steps:")
        for s in sorted(procedure.steps, key=lambda x: x.order):
            parts.append(f"  {s.order}. {s.action}")
        if procedure.postconditions:
            parts.append("Postconditions:")
            for pc in procedure.postconditions:
                parts.append(f"- {pc}")

    if golden.vocabulary:
        parts.append("\n# Vocabulary")
        for v in golden.vocabulary[:20]:
            parts.append(f"- {v.term}: {v.definition}")

    if golden.anti_patterns:
        parts.append("\n# Anti-patterns to avoid")
        for a in golden.anti_patterns[:10]:
            parts.append(f"- [{a.id}] {a.description} — instead: {a.correction}")

    parts.append(
        "\n# Behaviour\n"
        "- Stay strictly within the role above.\n"
        "- Cite principle/procedure ids in square brackets when helpful.\n"
        "- Do not invent facts; if unknown, say so.\n"
        "- Ask a clarifying question only when essential."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_PLANNER_SYSTEM = """You are the PLANNER agent for an SFT trajectory.

Given a scenario, decide:
- which principles the assistant MUST demonstrate (subset of `available_principle_ids`),
- which procedure (if any) it should follow (must be one of `available_procedure_ids` or null),
- an `outline` (1-3 short bullets describing the intended conversation arc),
- the `max_turns` you want (1-6).

Return JSON only. Schema:
{
  "principle_ids": ["PRN-...", ...],
  "procedure_id": "PRC-..." | null,
  "outline": ["...", ...],
  "max_turns": int
}
""".strip()


def _planner_user_prompt(scenario: Scenario, golden: GoldenDocument) -> str:
    return (
        f"SCENARIO id={scenario.id} difficulty={scenario.difficulty}\n"
        f"PERSONA id={scenario.persona_id}\n"
        f"PROCEDURE anchor: {scenario.procedure_id or 'none'}\n"
        f"PRINCIPLE anchors: {scenario.principle_ids}\n"
        f"PROMPT_SEED: {scenario.prompt_seed}\n\n"
        f"available_principle_ids: {[p.id for p in golden.principles]}\n"
        f"available_procedure_ids: {[p.id for p in golden.procedures]}\n"
    )


async def _planner_step(
    scenario: Scenario, golden: GoldenDocument, client: LLMClient
) -> dict[str, Any]:
    """Decide outline + which principles/procedures must be demonstrated."""
    try:
        obj = await client.complete_json(
            system=_PLANNER_SYSTEM,
            user=_planner_user_prompt(scenario, golden),
            cache_system=True,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning("Planner LLM call failed (%s); using scenario defaults.", exc)
        obj = {}

    valid_pids = {p.id for p in golden.principles}
    valid_procs = {p.id for p in golden.procedures}

    raw_principles = obj.get("principle_ids") if isinstance(obj, dict) else None
    principle_ids = [
        pid for pid in (raw_principles or []) if isinstance(pid, str) and pid in valid_pids
    ] or list(scenario.principle_ids)

    raw_proc = obj.get("procedure_id") if isinstance(obj, dict) else None
    procedure_id: str | None
    if isinstance(raw_proc, str) and raw_proc in valid_procs:
        procedure_id = raw_proc
    else:
        procedure_id = scenario.procedure_id

    outline_raw = obj.get("outline") if isinstance(obj, dict) else None
    outline = [str(x) for x in (outline_raw or []) if isinstance(x, (str, int, float))]
    if not outline:
        outline = [f"Respond to: {scenario.prompt_seed}"]

    max_turns_raw = obj.get("max_turns") if isinstance(obj, dict) else None
    try:
        max_turns = int(max_turns_raw) if max_turns_raw is not None else 3
    except (TypeError, ValueError):
        max_turns = 3
    max_turns = max(1, min(6, max_turns))

    return {
        "principle_ids": principle_ids,
        "procedure_id": procedure_id,
        "outline": outline,
        "max_turns": max_turns,
    }


# ---------------------------------------------------------------------------
# User simulator
# ---------------------------------------------------------------------------


_USER_SIM_SYSTEM = """You are the USER-SIM agent.

You play a HUMAN talking to an AI agent. Stay in character based on the persona
description. You may:
- send the opening message (use the prompt_seed verbatim or lightly paraphrased),
- ask follow-up questions when the assistant's answer leaves something open,
- push back if the assistant misses a key concern,
- signal you're done by replying with `<END>` on its own line.

Output ONLY the message text you would send (no role labels, no JSON, no quotes).
""".strip()


def _user_sim_user_prompt(
    *,
    scenario: Scenario,
    persona: Persona | None,
    history: list[TrajectoryMessage],
    turn_index: int,
    max_turns: int,
) -> str:
    persona_block = (
        f"PERSONA: {persona.name} ({persona.expertise_level}, tone: {persona.tone})\n"
        f"PERSONA DETAIL: {persona.description}\n"
        if persona is not None
        else f"PERSONA id={scenario.persona_id} (no detail available)\n"
    )
    transcript_lines: list[str] = []
    for m in history:
        if m.role == "user":
            transcript_lines.append(f"USER: {m.content}")
        elif m.role == "assistant":
            transcript_lines.append(f"ASSISTANT: {m.content}")
    transcript = "\n".join(transcript_lines) or "(no transcript yet)"

    instruction = (
        "Send the opening user message now. Use the prompt_seed."
        if turn_index == 0
        else (
            f"Turn {turn_index + 1} of at most {max_turns}. "
            "Reply as the user. If the assistant has fully resolved the request, "
            "reply with `<END>` only."
        )
    )

    return (
        f"{persona_block}\n"
        f"SCENARIO difficulty: {scenario.difficulty}\n"
        f"PROMPT_SEED: {scenario.prompt_seed}\n\n"
        f"TRANSCRIPT SO FAR:\n{transcript}\n\n"
        f"{instruction}"
    )


async def _user_sim_turn(
    *,
    history: list[TrajectoryMessage],
    scenario: Scenario,
    golden: GoldenDocument,
    client: LLMClient,
    persona: Persona | None = None,
    turn_index: int = 0,
    max_turns: int = 3,
) -> str:
    """Produce the next user message. May return `<END>` to signal completion."""
    user_prompt = _user_sim_user_prompt(
        scenario=scenario,
        persona=persona,
        history=history,
        turn_index=turn_index,
        max_turns=max_turns,
    )
    try:
        text = await client.complete(
            system=_USER_SIM_SYSTEM,
            user=user_prompt,
            cache_system=False,
            temperature=0.8,
        )
    except Exception as exc:
        logger.warning("User-sim LLM call failed (%s); ending conversation.", exc)
        return "<END>"
    return text.strip() or "<END>"


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------


async def _assistant_turn(
    *,
    history: list[TrajectoryMessage],
    system_prompt: str,
    client: LLMClient,
) -> str:
    """Produce the next assistant message given the role's system prompt."""
    api_messages: list[LLMMessage] = []
    for m in history:
        if m.role in ("user", "assistant"):
            api_messages.append(LLMMessage(role=m.role, content=m.content))
    if not api_messages or api_messages[-1].role != "user":
        # Defensive: assistant should only speak after a user message.
        return ""
    try:
        text = await client.complete_messages(
            system=system_prompt,
            messages=api_messages,
            cache_system=True,
            temperature=0.7,
        )
    except Exception as exc:
        logger.warning("Assistant LLM call failed (%s); emitting empty turn.", exc)
        return ""
    return text.strip()


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


_CRITIC_SYSTEM = """You are the CRITIC agent. Score a trajectory between an AI agent and a user.

You receive: the assistant's system prompt (identity + selected principles/procedure),
and the full transcript. Score 0.0 to 1.0 along:
- principle adherence,
- no hallucination beyond the golden context,
- format/structure quality,
- persona-appropriate user simulation.

Return JSON only. Schema:
{
  "score": float,    // 0.0-1.0
  "notes": "1-3 sentences explaining the score"
}
""".strip()


def _critic_user_prompt(system_prompt: str, messages: list[TrajectoryMessage]) -> str:
    lines: list[str] = ["SYSTEM PROMPT:", system_prompt, "", "TRANSCRIPT:"]
    for m in messages:
        if m.role == "user":
            lines.append(f"USER: {m.content}")
        elif m.role == "assistant":
            lines.append(f"ASSISTANT: {m.content}")
        elif m.role == "tool":
            lines.append(f"TOOL[{m.name or 'unknown'}]: {m.content}")
    return "\n".join(lines)


async def _critic_score(
    trajectory: Trajectory, golden: GoldenDocument, client: LLMClient
) -> tuple[float, str]:
    """Score the trajectory. Returns (score in [0,1], notes)."""
    _ = golden  # currently unused; system prompt already encodes the relevant golden content
    try:
        obj = await client.complete_json(
            system=_CRITIC_SYSTEM,
            user=_critic_user_prompt(trajectory.system, trajectory.messages),
            cache_system=True,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("Critic LLM call failed (%s); scoring 0.0.", exc)
        return 0.0, f"critic-error: {exc}"

    try:
        score = float(obj.get("score", 0.0)) if isinstance(obj, dict) else 0.0
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    notes = ""
    if isinstance(obj, dict):
        notes = str(obj.get("notes", "")).strip()
    return score, notes


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _trajectory_id(scenario_id: str) -> str:
    return f"TRJ-{scenario_id}"


def _persona_lookup(personas: list[Persona] | None, persona_id: str) -> Persona | None:
    if not personas:
        return None
    for p in personas:
        if p.id == persona_id:
            return p
    return None


def _is_end_signal(text: str) -> bool:
    s = text.strip()
    return s == "<END>" or s.upper() == "<END>" or s.endswith("<END>")


async def build_trajectory(
    scenario: Scenario,
    golden: GoldenDocument,
    client: LLMClient,
    max_turns: int = 6,
    *,
    personas: list[Persona] | None = None,
    config: SFTConfig | None = None,
) -> Trajectory:
    """Run the 4-agent rollout for one scenario and return a graded Trajectory.

    `max_turns` bounds the number of *user* turns (so the conversation has at most
    `2 * max_turns` messages excluding the system prompt). The planner may request
    a smaller bound; the smaller of the two is used.
    """
    cfg = config or SFTConfig()
    plan = await _planner_step(scenario, golden, client)
    bound = min(max_turns, int(plan.get("max_turns", max_turns)))
    bound = max(1, bound)

    # Build a scenario-effective copy reflecting the planner's choices, used for
    # system prompt construction.
    effective_scenario = scenario.model_copy(
        update={
            "principle_ids": plan["principle_ids"],
            "procedure_id": plan["procedure_id"],
        }
    )
    system_prompt = build_system_prompt(golden, effective_scenario)
    persona = _persona_lookup(personas, scenario.persona_id)

    messages: list[TrajectoryMessage] = []

    for turn_index in range(bound):
        user_text = await _user_sim_turn(
            history=messages,
            scenario=scenario,
            golden=golden,
            client=client,
            persona=persona,
            turn_index=turn_index,
            max_turns=bound,
        )
        if turn_index > 0 and _is_end_signal(user_text):
            break
        # Strip a trailing `<END>` token that may be appended to a final message.
        if _is_end_signal(user_text):
            user_text = user_text.replace("<END>", "").strip()
            if not user_text:
                break
        messages.append(TrajectoryMessage(role="user", content=user_text))

        assistant_text = await _assistant_turn(
            history=messages, system_prompt=system_prompt, client=client
        )
        messages.append(TrajectoryMessage(role="assistant", content=assistant_text))

    procedure_ids = [effective_scenario.procedure_id] if effective_scenario.procedure_id else []
    tags: dict[str, Any] = {
        "procedure_ids": procedure_ids,
        "principle_ids": list(effective_scenario.principle_ids),
        "persona": scenario.persona_id,
        "difficulty": scenario.difficulty,
    }

    trajectory = Trajectory(
        id=_trajectory_id(scenario.id),
        scenario_id=scenario.id,
        system=system_prompt,
        messages=messages,
        tags=tags,
    )

    score, notes = await _critic_score(trajectory, golden, client)
    trajectory.quality_score = score
    trajectory.critic_notes = notes

    # Repair loop: re-roll the last assistant turn if any assistant turn exists.
    attempts = 0
    repair_notes: list[str] = []
    while (
        trajectory.quality_score is not None
        and trajectory.quality_score < cfg.critic_min_score
        and attempts < cfg.max_repair_attempts
        and any(m.role == "assistant" for m in trajectory.messages)
    ):
        attempts += 1
        # Locate the last assistant message.
        last_assist_idx: int | None = None
        for idx in range(len(trajectory.messages) - 1, -1, -1):
            if trajectory.messages[idx].role == "assistant":
                last_assist_idx = idx
                break
        if last_assist_idx is None:
            break
        history_before = trajectory.messages[:last_assist_idx]
        new_text = await _assistant_turn(
            history=history_before, system_prompt=system_prompt, client=client
        )
        trajectory.messages[last_assist_idx] = TrajectoryMessage(
            role="assistant", content=new_text
        )
        new_score, new_notes = await _critic_score(trajectory, golden, client)
        repair_notes.append(f"repair#{attempts} score={new_score:.2f} :: {new_notes}")
        trajectory.quality_score = new_score
        trajectory.critic_notes = new_notes

    if repair_notes:
        trajectory.critic_notes = (
            (trajectory.critic_notes or "") + " | " + " ; ".join(repair_notes)
        ).strip(" |")
    return trajectory


__all__ = [
    "build_system_prompt",
    "build_trajectory",
]
