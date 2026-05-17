"""Swarm orchestration: personas → scenarios → trajectories → dedup → train/eval split.

Streaming model
---------------
The swarm used to ``asyncio.gather`` thousands of trajectory coroutines and only
then dedup/filter them. With ``n_trajectories=10_000`` that pinned ~10k Tasks +
multi-megabyte transcripts in memory before any disk write could happen, and a
single failure cancelled the whole batch.

The new flow is a bounded producer/consumer:

* A queue of pending train scenarios.
* ``N = config.llm.max_concurrent`` worker coroutines pull scenarios, call
  :func:`build_trajectory`, and push the result (or the raised exception) onto
  a result queue.
* The producer drains the result queue, applies the per-trajectory dedup
  signature check and the critic-score filter inline, and yields survivors to
  the caller via :func:`stream_swarm`.

A worker raising never cancels its peers — only that one trajectory is lost.
Peak resident memory is ``O(max_concurrent)`` trajectories instead of
``O(n_trajectories)``.

``run_swarm`` remains the convenience wrapper that drains
:func:`stream_swarm` into ``(train, eval)`` lists for callers that prefer the
legacy shape (the autonomous runner, existing tests).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from typing import Literal

from gyroscope.core.config import SFTConfig
from gyroscope.core.llm import LLMClient
from gyroscope.core.models import GoldenDocument, Persona, Scenario, Trajectory
from gyroscope.sft.personas import generate_personas
from gyroscope.sft.scenarios import generate_scenarios
from gyroscope.sft.trajectory import build_trajectory

logger = logging.getLogger(__name__)


SplitLabel = Literal["train", "eval"]

# Sentinel pushed onto the result queue when a worker exits, so the producer
# knows how many workers have finished without polling task state.
_WORKER_DONE = object()


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


def _signature(traj: Trajectory, scenario: Scenario | None) -> set[str]:
    seed = scenario.prompt_seed if scenario is not None else ""
    return _token_set(seed + " " + _first_user_message(traj))


def _is_duplicate(sig: set[str], pool: list[set[str]], threshold: float) -> bool:
    if threshold <= 0:
        return False
    return any(_jaccard(sig, existing) >= threshold for existing in pool)


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
        sig = _signature(traj, scen)
        if _is_duplicate(sig, kept_sigs, threshold):
            continue
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


def _passes_quality(traj: Trajectory, min_score: float) -> bool:
    return traj.quality_score is not None and traj.quality_score >= min_score


def _resolve_worker_count(client: LLMClient, scenario_count: int) -> int:
    """Pick the worker pool size.

    Honours ``client.config.llm.max_concurrent`` when available (production
    ``LLMClient``); falls back to a small default for test stubs that don't
    expose a config object. Capped at the number of scenarios so we don't
    spawn idle workers.
    """
    default = 16
    try:
        configured = int(client.config.llm.max_concurrent)
    except AttributeError:
        configured = default
    n = max(1, configured)
    return min(n, max(1, scenario_count))


async def _build_one(
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


async def _worker(
    name: str,
    pending: asyncio.Queue[Scenario | None],
    results: asyncio.Queue[tuple[Scenario, Trajectory | BaseException] | object],
    golden: GoldenDocument,
    client: LLMClient,
    config: SFTConfig,
    personas: list[Persona],
) -> None:
    """Pull scenarios off ``pending`` and push outcomes onto ``results``.

    Worker-local exceptions are pushed onto the result queue rather than
    raised so one failing scenario cannot cancel the rest of the swarm.
    The worker exits when it pulls a ``None`` sentinel.
    """
    while True:
        scenario = await pending.get()
        try:
            if scenario is None:
                return
            try:
                traj = await _build_one(scenario, golden, client, config, personas)
            except BaseException as exc:
                logger.warning(
                    "worker %s: build_trajectory failed for %s: %s",
                    name,
                    scenario.id,
                    exc,
                )
                await results.put((scenario, exc))
            else:
                await results.put((scenario, traj))
        finally:
            pending.task_done()


async def _drain_split_concurrently(
    scenarios: list[Scenario],
    split_label: SplitLabel,
    golden: GoldenDocument,
    client: LLMClient,
    config: SFTConfig,
    personas: list[Persona],
    scenario_index: dict[str, Scenario],
) -> AsyncIterator[tuple[Trajectory, SplitLabel]]:
    """Yield surviving (trajectory, split_label) pairs for one split.

    Implements the bounded-worker pattern: a single pending queue feeds
    ``N`` worker coroutines that push outcomes onto a result queue; the
    producer dedups + filters inline and yields survivors. Memory footprint
    is ``O(max_concurrent)`` regardless of ``len(scenarios)``.
    """
    if not scenarios:
        return

    n_workers = _resolve_worker_count(client, len(scenarios))
    pending: asyncio.Queue[Scenario | None] = asyncio.Queue()
    results: asyncio.Queue[tuple[Scenario, Trajectory | BaseException] | object] = asyncio.Queue()

    for s in scenarios:
        pending.put_nowait(s)
    for _ in range(n_workers):
        pending.put_nowait(None)  # one stop sentinel per worker

    workers = [
        asyncio.create_task(
            _worker(f"{split_label}-{i}", pending, results, golden, client, config, personas),
            name=f"sft-swarm-{split_label}-worker-{i}",
        )
        for i in range(n_workers)
    ]

    # Notify the producer when every worker has finished. Running this as a
    # task means we don't have to join workers inline — we just count
    # ``_WORKER_DONE`` markers off the result queue.
    async def _signal_completion() -> None:
        for w in workers:
            try:
                await w
            except BaseException as exc:
                logger.error("swarm worker raised unexpectedly: %s", exc)
            await results.put(_WORKER_DONE)

    completion_task = asyncio.create_task(
        _signal_completion(), name=f"sft-swarm-{split_label}-completion"
    )

    kept_sigs: list[set[str]] = []
    workers_remaining = n_workers
    try:
        while workers_remaining > 0:
            item = await results.get()
            if item is _WORKER_DONE:
                workers_remaining -= 1
                continue
            assert isinstance(item, tuple)
            scenario, outcome = item
            if isinstance(outcome, BaseException):
                # Already logged inside the worker; skip the trajectory.
                continue
            if not _passes_quality(outcome, config.critic_min_score):
                continue
            sig = _signature(outcome, scenario_index.get(outcome.scenario_id, scenario))
            if _is_duplicate(sig, kept_sigs, config.dedup_threshold):
                continue
            kept_sigs.append(sig)
            yield outcome, split_label
    finally:
        # Best-effort cleanup if the consumer abandons the iterator early.
        for w in workers:
            if not w.done():
                w.cancel()
        if not completion_task.done():
            completion_task.cancel()
        # Surface any pending exceptions without raising — workers already
        # logged the underlying cause.
        for w in workers:
            with _suppress_cancelled():
                await _await_silent(w)
        with _suppress_cancelled():
            await _await_silent(completion_task)


class _suppress_cancelled:
    """``contextlib.suppress(CancelledError)`` without the import overhead."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> bool:
        return exc_type is not None and issubclass(exc_type, asyncio.CancelledError)


async def _await_silent(task: asyncio.Task[None]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        raise
    except BaseException:  # best-effort cleanup; workers already logged
        pass


async def stream_swarm(
    golden: GoldenDocument,
    client: LLMClient,
    config: SFTConfig,
) -> AsyncIterator[tuple[Trajectory, SplitLabel]]:
    """Stream surviving (trajectory, split_label) pairs as they are produced.

    Replaces the old "build everything, then dedup, then filter" pipeline with
    a bounded producer/consumer. Survivors are yielded as soon as each
    worker finishes a trajectory and it clears the dedup + quality gates, so
    the caller can write them to disk without buffering the full dataset.

    The train split is yielded first (in worker-completion order), then the
    eval split. Yielding in two phases keeps the dedup signature pools
    procedure-disjoint, matching the legacy :func:`run_swarm` semantics.
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
        return

    train_procs, eval_procs = _procedure_split(scenarios, config.eval_holdout_fraction)
    train_scenarios = [s for s in scenarios if s.procedure_id in train_procs]
    eval_scenarios = [s for s in scenarios if s.procedure_id in eval_procs]
    scenario_index: dict[str, Scenario] = {s.id: s for s in scenarios}

    # Prime the build_stable_system_prefix cache once so every worker hands
    # the same memoized string to the Anthropic client (cache-friendly +
    # avoids per-trajectory string concatenation).
    from gyroscope.sft.trajectory import build_stable_system_prefix

    build_stable_system_prefix(golden)

    train_yielded = 0
    async for item in _drain_split_concurrently(
        train_scenarios,
        "train",
        golden,
        client,
        config,
        personas,
        scenario_index,
    ):
        train_yielded += 1
        yield item

    eval_yielded = 0
    async for item in _drain_split_concurrently(
        eval_scenarios,
        "eval",
        golden,
        client,
        config,
        personas,
        scenario_index,
    ):
        eval_yielded += 1
        yield item

    logger.info(
        "swarm complete: scenarios=%d train_kept=%d eval_kept=%d",
        len(scenarios),
        train_yielded,
        eval_yielded,
    )


async def run_swarm(
    golden: GoldenDocument,
    client: LLMClient,
    config: SFTConfig,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Run the swarm and return (train_trajectories, eval_trajectories).

    Convenience wrapper around :func:`stream_swarm` that drains the iterator
    into two lists. Existing callers (the autonomous runner, plus tests
    written before the streaming refactor) keep the legacy tuple shape.
    Prefer :func:`stream_swarm` for new code that wants to bound memory.
    """
    train: list[Trajectory] = []
    evals: list[Trajectory] = []
    async for traj, split in stream_swarm(golden, client, config):
        if split == "train":
            train.append(traj)
        else:
            evals.append(traj)
    return train, evals


__all__ = ["run_swarm", "semantic_dedup", "stream_swarm"]
