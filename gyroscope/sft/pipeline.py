"""High-level pipeline that writes `sft.jsonl` and `eval.jsonl` to disk.

The pipeline streams surviving train trajectories straight to disk via
:func:`stream_swarm`. Each row is rendered and appended to the open
``sft.jsonl`` handle as the worker pool produces it, so peak resident
state for the train split stays at ``O(max_concurrent)`` — no full-list
buffering. The eval split is small (procedure-disjoint hold-out) so we
keep it in memory in order to run the leakage check + :class:`EvalPipeline.write`
flow unchanged.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from gyroscope.core.config import SFTConfig
from gyroscope.core.llm import LLMClient
from gyroscope.core.models import GoldenDocument, Trajectory
from gyroscope.eval.pipeline import EvalPipeline
from gyroscope.sft.formats import render
from gyroscope.sft.swarm import stream_swarm

logger = logging.getLogger(__name__)


class SFTPipeline:
    """Phase 3 entry point. Runs the swarm and serialises the output dataset."""

    async def run(
        self,
        golden: GoldenDocument,
        output_dir: Path | str,
        client: LLMClient,
        config: SFTConfig,
    ) -> tuple[Path, Path]:
        """Generate the SFT dataset and write `sft.jsonl` + `eval.jsonl`.

        Returns the (train_path, eval_path) tuple.

        Train trajectories are written to ``sft.jsonl`` as they stream out of
        :func:`stream_swarm`; only the eval split is buffered (small held-out
        set) so :class:`EvalPipeline` can run the procedure-level leakage
        check against the full train list.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "sft.jsonl"

        # Keep a small list of train trajectory metadata for the leakage check
        # against eval (the procedure-disjoint guarantee — only tags +
        # scenario_id are needed). Full train trajectories are streamed straight
        # to disk and never held resident.
        train_meta: list[Trajectory] = []
        train_tags_counter: Counter[str] = Counter()
        train_procs_counter: Counter[str] = Counter()
        evals: list[Trajectory] = []
        n_train = 0

        # Open the sink once, append each rendered row as the worker pool
        # yields a survivor. Peak resident trajectories stays O(max_concurrent)
        # rather than O(n_train).
        with train_path.open("w", encoding="utf-8") as fh:
            async for traj, split in stream_swarm(golden, client, config):
                if split == "train":
                    fh.write(json.dumps(render(traj, config.output_format), ensure_ascii=False))
                    fh.write("\n")
                    n_train += 1
                    # Track only the lightweight bookkeeping we need afterwards.
                    train_meta.append(_strip_to_metadata(traj))
                    train_tags_counter[str(traj.tags.get("difficulty", "unknown"))] += 1
                    for pid in traj.tags.get("procedure_ids", []) or ["(none)"]:
                        train_procs_counter[str(pid)] += 1
                else:
                    evals.append(traj)

        self._log_counter("train", "difficulty", train_tags_counter, n_train)
        self._log_counter("train", "procedure", train_procs_counter, n_train)
        self._log_distribution("eval", evals)

        # Route the eval split through EvalPipeline so the procedure-level
        # leakage check actually runs in production (not just unit tests).
        eval_pipeline = EvalPipeline(output_format=config.output_format)
        eval_path = eval_pipeline.write(evals, train_meta, out_dir, strict=config.eval_strict)

        logger.info(
            "wrote %d train rows to %s and %d eval rows to %s",
            n_train,
            train_path,
            len(evals),
            eval_path,
        )
        return train_path, eval_path

    @staticmethod
    def _log_distribution(label: str, trajectories: Iterable[Trajectory]) -> None:
        materialised = list(trajectories) if not isinstance(trajectories, list) else trajectories
        if not materialised:
            logger.info("%s split is empty", label)
            return
        by_difficulty: Counter[str] = Counter()
        by_procedure: Counter[str] = Counter()
        for t in materialised:
            difficulty = str(t.tags.get("difficulty", "unknown"))
            by_difficulty[difficulty] += 1
            for pid in t.tags.get("procedure_ids", []) or ["(none)"]:
                by_procedure[str(pid)] += 1
        logger.info(
            "%s difficulty distribution: %s",
            label,
            dict(sorted(by_difficulty.items())),
        )
        logger.info(
            "%s procedure coverage: %s",
            label,
            dict(sorted(by_procedure.items())),
        )

    @staticmethod
    def _log_counter(split_label: str, axis: str, counts: Counter[str], total: int) -> None:
        if total == 0:
            logger.info("%s split is empty", split_label)
            return
        logger.info("%s %s distribution: %s", split_label, axis, dict(sorted(counts.items())))


def _strip_to_metadata(traj: Trajectory) -> Trajectory:
    """Return a Trajectory carrying only the tags/scenario_id the leakage
    check needs. We zero out the message list so we don't keep multi-KB
    transcripts resident just to detect procedure overlap."""
    return traj.model_copy(update={"messages": [], "system": ""})


__all__ = ["SFTPipeline"]
