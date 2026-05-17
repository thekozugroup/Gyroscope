"""Short-lived 4-agent rollout that produces a single Trajectory.

Agents:
- Planner   — picks an outline and target principle/procedure ids (optional).
- User-sim  — plays the user (persona-consistent, multi-turn).
- Assistant — plays the role using the golden doc as context.
- Critic    — scores principle adherence, no hallucination, format.

A single repair pass (re-roll the last assistant turn) is attempted when the
critic score falls below `SFTConfig.critic_min_score`, up to
`SFTConfig.max_repair_attempts` times.

System-prompt design (cache-friendly):
    The assistant and critic share a *stable* system prefix that is identical
    across every scenario sharing the same ``GoldenDocument`` — this lets
    Anthropic prompt caching reuse the cached prefix for every assistant /
    critic call (cache hits across ~60k calls per 5k-trajectory run). The
    per-scenario principle/procedure selection is injected into the *user*
    message instead, where caching does not matter.
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


def _selected_procedure(golden: GoldenDocument, scenario: Scenario) -> Procedure | None:
    if not scenario.procedure_id:
        return None
    for p in golden.procedures:
        if p.id == scenario.procedure_id:
            return p
    return None


def build_stable_system_prefix(golden: GoldenDocument) -> str:
    """Construct the assistant's *scenario-independent* system prefix.

    This string is identical for every trajectory that shares ``golden`` so it
    can hit the Anthropic prompt cache across every assistant + critic LLM
    call in a run. Per-scenario selections (selected principles / procedure)
    are NOT included here — they go in the user turn via
    :func:`build_scenario_suffix`.
    """
    parts: list[str] = []
    parts.append(f"# Identity\nYou are: {golden.identity.role}.")
    parts.append(golden.identity.description)
    parts.append(f"\n# Mission\n{golden.identity.mission}")

    if golden.principles:
        parts.append("\n# Principles (full list)")
        for p in golden.principles:
            parts.append(f"- [{p.id}] {p.statement}")

    if golden.procedures:
        parts.append("\n# Procedures (full list)")
        for procedure in golden.procedures:
            parts.append(f"\n## [{procedure.id}] {procedure.name}")
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
        for v in golden.vocabulary:
            parts.append(f"- {v.term}: {v.definition}")

    if golden.anti_patterns:
        parts.append("\n# Anti-patterns to avoid")
        for a in golden.anti_patterns:
            parts.append(f"- [{a.id}] {a.description} — instead: {a.correction}")

    parts.append(
        "\n# Behaviour\n"
        "- Stay strictly within the role above.\n"
        "- Cite principle/procedure ids in square brackets when helpful.\n"
        "- Do not invent facts; if unknown, say so.\n"
        "- Ask a clarifying question only when essential."
    )
    return "\n".join(parts)


def build_scenario_suffix(scenario: Scenario, golden: GoldenDocument) -> str:
    """Per-scenario focusing block prepended to the first user turn.

    Lists which principle ids / procedure id this trajectory should
    demonstrate. Kept short so it doesn't dilute the cached prefix.
    """
    principles = _select_principles(golden, scenario)
    procedure = _selected_procedure(golden, scenario)
    lines: list[str] = ["# Scenario focus"]
    if principles:
        lines.append("Principles to demonstrate this turn:")
        for p in principles:
            lines.append(f"- [{p.id}] {p.statement}")
    if procedure is not None:
        lines.append(f"Procedure to follow: [{procedure.id}] {procedure.name}")
    if len(lines) == 1:
        lines.append("(no specific focus — answer naturally within the role.)")
    return "\n".join(lines)


def build_system_prompt(golden: GoldenDocument, scenario: Scenario) -> str:
    """Backwards-compatible composition of stable prefix + scenario suffix.

    New code paths prefer :func:`build_stable_system_prefix` (cached) +
    :func:`build_scenario_suffix` (injected into the first user turn) so
    Anthropic prompt caching is effective across scenarios. This combined
    form is retained for the recorded ``Trajectory.system`` field and for
    the legacy public API.
    """
    return build_stable_system_prefix(golden) + "\n\n" + build_scenario_suffix(scenario, golden)


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
    scenario: Scenario,
    golden: GoldenDocument,
    client: LLMClient,
    config: SFTConfig | None = None,
) -> dict[str, Any]:
    """Decide outline + which principles/procedures must be demonstrated."""
    cfg = config or SFTConfig()
    try:
        obj = await client.complete_json(
            system=_PLANNER_SYSTEM,
            user=_planner_user_prompt(scenario, golden),
            cache_system=True,
            temperature=cfg.temperature_planner,
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
    config: SFTConfig | None = None,
) -> str:
    """Produce the next user message. May return `<END>` to signal completion."""
    cfg = config or SFTConfig()
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
            temperature=cfg.temperature_user_sim,
        )
    except Exception as exc:
        logger.warning("User-sim LLM call failed (%s); ending conversation.", exc)
        return "<END>"
    return text.strip() or "<END>"


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------


def _inject_scenario_suffix(
    history: list[TrajectoryMessage], scenario_suffix: str
) -> list[LLMMessage]:
    """Build the assistant API message list, prepending the per-scenario focus
    block to the *first* user message so the system prompt itself stays
    byte-identical across scenarios (cache hit territory)."""
    api_messages: list[LLMMessage] = []
    first_user_seen = False
    for m in history:
        if m.role not in ("user", "assistant"):
            continue
        content = m.content
        if not first_user_seen and m.role == "user":
            content = f"{scenario_suffix}\n\n{content}" if scenario_suffix else content
            first_user_seen = True
        api_messages.append(LLMMessage(role=m.role, content=content))
    return api_messages


async def _assistant_turn(
    *,
    history: list[TrajectoryMessage],
    system_prompt: str,
    client: LLMClient,
    scenario_suffix: str = "",
    config: SFTConfig | None = None,
) -> str:
    """Produce the next assistant message given the role's system prompt.

    ``system_prompt`` should be the *stable* prefix (same for every scenario
    that shares a golden document); ``scenario_suffix`` is folded into the
    first user message so prompt caching can re-use the prefix across calls.
    """
    cfg = config or SFTConfig()
    api_messages = _inject_scenario_suffix(history, scenario_suffix)
    if not api_messages or api_messages[-1].role != "user":
        # Defensive: assistant should only speak after a user message.
        return ""
    try:
        text = await client.complete_messages(
            system=system_prompt,
            messages=api_messages,
            cache_system=True,
            temperature=cfg.temperature_assistant,
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


def _critic_user_prompt(scenario_suffix: str, messages: list[TrajectoryMessage]) -> str:
    """Build the critic *user* message.

    The stable critic system prompt is the SAME for every scenario (so it
    hits the prompt cache). The per-scenario context — the focusing block
    listing the selected principle / procedure ids — is folded into the
    user message instead. This keeps the system prefix byte-identical
    across all critic calls in a run.
    """
    lines: list[str] = []
    if scenario_suffix:
        lines.append("SCENARIO CONTEXT:")
        lines.append(scenario_suffix)
        lines.append("")
    lines.append("TRANSCRIPT:")
    for m in messages:
        if m.role == "user":
            lines.append(f"USER: {m.content}")
        elif m.role == "assistant":
            lines.append(f"ASSISTANT: {m.content}")
        elif m.role == "tool":
            lines.append(f"TOOL[{m.name or 'unknown'}]: {m.content}")
    return "\n".join(lines)


async def _critic_score(
    trajectory: Trajectory,
    golden: GoldenDocument,
    client: LLMClient,
    *,
    stable_system_prefix: str | None = None,
    scenario_suffix: str = "",
    config: SFTConfig | None = None,
) -> tuple[float, str]:
    """Score the trajectory. Returns (score in [0,1], notes).

    The critic uses the same stable-prefix / scenario-suffix split as the
    assistant turn, so its system prompt is identical across every scenario
    sharing the same ``GoldenDocument`` (cache hit territory).
    """
    cfg = config or SFTConfig()
    critic_system = _CRITIC_SYSTEM
    if stable_system_prefix is not None:
        # Append the role's full identity / principles / procedures to the
        # critic system prompt so the cached prefix is content-rich and the
        # critic doesn't have to re-derive context from the transcript.
        critic_system = _CRITIC_SYSTEM + "\n\n" + stable_system_prefix
    _ = golden  # golden content already lives in stable_system_prefix
    try:
        obj = await client.complete_json(
            system=critic_system,
            user=_critic_user_prompt(scenario_suffix, trajectory.messages),
            cache_system=True,
            temperature=cfg.temperature_critic,
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


def _plan_from_scenario(scenario: Scenario, max_turns: int) -> dict[str, Any]:
    """Build a planner-output-shaped dict directly from the scenario.

    Used when ``SFTConfig.use_planner`` is False — the deterministic scenario
    generator already picks the principle ids and procedure, so the planner
    LLM call is pure overhead in steady-state runs.
    """
    return {
        "principle_ids": list(scenario.principle_ids),
        "procedure_id": scenario.procedure_id,
        "outline": [f"Respond to: {scenario.prompt_seed}"],
        "max_turns": max_turns,
    }


async def build_trajectory(
    scenario: Scenario,
    golden: GoldenDocument,
    client: LLMClient,
    max_turns: int | None = None,
    *,
    personas: list[Persona] | None = None,
    config: SFTConfig | None = None,
) -> Trajectory:
    """Run the (up to) 4-agent rollout for one scenario and return a graded
    Trajectory.

    ``max_turns`` bounds the number of *user* turns (so the conversation has
    at most ``2 * max_turns`` messages excluding the system prompt). When the
    planner runs it may request a smaller bound; the smaller of the two is
    used. When ``max_turns`` is None the value falls back to
    ``config.max_turns`` (which itself defaults to 6).
    """
    cfg = config or SFTConfig()
    effective_max_turns = cfg.max_turns if max_turns is None else max_turns

    if cfg.use_planner:
        plan = await _planner_step(scenario, golden, client, config=cfg)
    else:
        plan = _plan_from_scenario(scenario, effective_max_turns)

    bound = min(effective_max_turns, int(plan.get("max_turns", effective_max_turns)))
    bound = max(1, bound)

    # Build a scenario-effective copy reflecting the planner's choices, used for
    # the per-scenario focus block.
    effective_scenario = scenario.model_copy(
        update={
            "principle_ids": plan["principle_ids"],
            "procedure_id": plan["procedure_id"],
        }
    )
    # Cache-friendly split: the *stable* prefix is identical across every
    # scenario that shares ``golden``; the per-scenario focus block is folded
    # into the first user message so the system prompt stays byte-identical
    # and Anthropic prompt caching can serve every assistant/critic call.
    stable_prefix = build_stable_system_prefix(golden)
    scenario_suffix = build_scenario_suffix(effective_scenario, golden)
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
            config=cfg,
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
            history=messages,
            system_prompt=stable_prefix,
            client=client,
            scenario_suffix=scenario_suffix,
            config=cfg,
        )
        messages.append(TrajectoryMessage(role="assistant", content=assistant_text))

    procedure_ids = [effective_scenario.procedure_id] if effective_scenario.procedure_id else []
    tags: dict[str, Any] = {
        "procedure_ids": procedure_ids,
        "principle_ids": list(effective_scenario.principle_ids),
        "persona": scenario.persona_id,
        "difficulty": scenario.difficulty,
    }

    # Record the *combined* prefix + suffix in Trajectory.system so the
    # downstream writers (sharegpt/chatml/alpaca) and consumers see the full
    # in-context system prompt, even though over the wire we shipped the two
    # halves separately for caching.
    full_system = stable_prefix + "\n\n" + scenario_suffix

    trajectory = Trajectory(
        id=_trajectory_id(scenario.id),
        scenario_id=scenario.id,
        system=full_system,
        messages=messages,
        tags=tags,
    )

    score, notes = await _critic_score(
        trajectory,
        golden,
        client,
        stable_system_prefix=stable_prefix,
        scenario_suffix=scenario_suffix,
        config=cfg,
    )
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
            history=history_before,
            system_prompt=stable_prefix,
            client=client,
            scenario_suffix=scenario_suffix,
            config=cfg,
        )
        trajectory.messages[last_assist_idx] = TrajectoryMessage(role="assistant", content=new_text)
        new_score, new_notes = await _critic_score(
            trajectory,
            golden,
            client,
            stable_system_prefix=stable_prefix,
            scenario_suffix=scenario_suffix,
            config=cfg,
        )
        repair_notes.append(f"repair#{attempts} score={new_score:.2f} :: {new_notes}")
        trajectory.quality_score = new_score
        trajectory.critic_notes = new_notes

    if repair_notes:
        trajectory.critic_notes = (
            (trajectory.critic_notes or "") + " | " + " ; ".join(repair_notes)
        ).strip(" |")
    return trajectory


__all__ = [
    "build_scenario_suffix",
    "build_stable_system_prefix",
    "build_system_prompt",
    "build_trajectory",
]
