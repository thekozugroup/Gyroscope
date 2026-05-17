"""Swarm orchestration: personas → scenarios → trajectories → dedup → train/eval split."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable

from gyroscope.core.config import SFTConfig
from gyroscope.core.llm import LLMClient
from gyroscope.core.models import GoldenDocument, Persona, Scenario, Trajectory
from gyroscope.sft.personas import generate_personas
from gyroscope.sft.scenarios import generate_scenarios
from gyroscope.sft.trajectory import build_trajectory

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _first_user_message(traj: Trajectory) -> str:
    for m in traj.messages:
        if m.role == "user":
            return m.content
    return ""


def semantic_dedup(
    trajectories: list[Trajectory],
    scenarios: dict[str, Scenario],
    threshold: float,
) -> list[Trajectory]:
    """Drop trajectories that are near-duplicates of an earlier one.

    Similarity is the token-set Jaccard over `scenario.prompt_seed + first_user_message`.
    Trajectories are inspected in input order; the first occurrence wins.
    """
    if threshold <= 0:
        return list(trajectories)

    kept: list[Trajectory] = []
    kept_sigs: list[set[str]] = []
    for traj in trajectories:
        scen = scenarios.get(traj.scenario_id)
        seed = scen.prompt_seed if scen is not None else ""
        sig = _token_set(seed + " " + _first_user_message(traj))
        duplicate = False
        for existing in kept_sigs:
            if _jaccard(sig, existing) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(traj)
            kept_sigs.append(sig)
    return kept


def _procedure_split(
    scenarios: list[Scenario], holdout_fraction: float
) -> tuple[set[str | None], set[str | None]]:
    """Split *procedure ids* (not scenarios) so eval procedures don't appear in train.

    Returns (train_procedure_ids, eval_procedure_ids). `None` (no-procedure scenarios)
    always goes to train to avoid concentrating it in eval.
    """
    proc_ids: list[str | None] = []
    seen: set[str | None] = set()
    for s in scenarios:
        if s.procedure_id in seen:
            continue
        seen.add(s.procedure_id)
        proc_ids.append(s.procedure_id)

    real_procs = [pid for pid in proc_ids if pid is not None]
    if not real_procs:
        return set(proc_ids), set()

    n_eval = max(1, round(len(real_procs) * holdout_fraction))
    n_eval = min(n_eval, max(1, len(real_procs) - 1))  # keep at least one for train
    # Deterministic split: last `n_eval` real procs (in encounter order) become eval.
    eval_procs: set[str | None] = set(real_procs[-n_eval:])
    train_procs: set[str | None] = {pid for pid in proc_ids if pid not in eval_procs}
    return train_procs, eval_procs


async def _bounded_build(
    scenario: Scenario,
    golden: GoldenDocument,
    client: LLMClient,
    config: SFTConfig,
    personas: list[Persona],
) -> Trajectory:
    return await build_trajectory(
        scenario,
        golden,
        client,
        max_turns=config.max_turns,
        personas=personas,
        config=config,
    )


def _drop_low_quality(trajectories: Iterable[Trajectory], min_score: float) -> list[Trajectory]:
    out: list[Trajectory] = []
    for t in trajectories:
        if t.quality_score is None or t.quality_score < min_score:
            continue
        out.append(t)
    return out


async def run_swarm(
    golden: GoldenDocument,
    client: LLMClient,
    config: SFTConfig,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Run the full swarm and return (train_trajectories, eval_trajectories).

    Steps:
      1. Generate personas.
      2. Generate stratified scenarios.
      3. Split procedures into train/eval (leakage prevention).
      4. Build trajectories concurrently (bounded by the LLM client semaphore).
      5. Semantic dedup over `prompt_seed + first_user_message`.
      6. Drop trajectories below the critic threshold (with repair attempts already spent).
    """
    personas = await generate_personas(
        golden, client, config.n_personas, temperature=config.temperature_persona
    )
    scenarios = await generate_scenarios(
        golden,
        personas,
        client,
        config.n_trajectories,
        config.difficulty_mix,
        temperature=config.temperature_scenario,
    )
    if not scenarios:
        return [], []

    train_procs, eval_procs = _procedure_split(scenarios, config.eval_holdout_fraction)
    train_scenarios = [s for s in scenarios if s.procedure_id in train_procs]
    eval_scenarios = [s for s in scenarios if s.procedure_id in eval_procs]

    # Build all trajectories concurrently — the LLMClient enforces the global cap.
    all_scenarios = train_scenarios + eval_scenarios
    coros = [_bounded_build(s, golden, client, config, personas) for s in all_scenarios]
    results = await asyncio.gather(*coros)

    by_id: dict[str, Scenario] = {s.id: s for s in all_scenarios}
    train_results: list[Trajectory] = []
    eval_results: list[Trajectory] = []
    train_ids = {s.id for s in train_scenarios}
    for traj in results:
        if traj.scenario_id in train_ids:
            train_results.append(traj)
        else:
            eval_results.append(traj)

    # Dedup within each split independently.
    train_deduped = semantic_dedup(train_results, by_id, config.dedup_threshold)
    eval_deduped = semantic_dedup(eval_results, by_id, config.dedup_threshold)

    # Drop sub-threshold trajectories (repair already attempted inside build_trajectory).
    train_final = _drop_low_quality(train_deduped, config.critic_min_score)
    eval_final = _drop_low_quality(eval_deduped, config.critic_min_score)

    logger.info(
        "swarm complete: scenarios=%d train_kept=%d eval_kept=%d",
        len(all_scenarios),
        len(train_final),
        len(eval_final),
    )
    return train_final, eval_final


__all__ = ["run_swarm", "semantic_dedup"]
