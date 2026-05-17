"""High-level pipeline that writes `sft.jsonl` and `eval.jsonl` to disk."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from gyroscope.core.config import SFTConfig
from gyroscope.core.io import write_jsonl
from gyroscope.core.llm import LLMClient
from gyroscope.core.models import GoldenDocument, Trajectory
from gyroscope.eval.pipeline import EvalPipeline
from gyroscope.sft.formats import render
from gyroscope.sft.swarm import run_swarm

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
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train, evals = await run_swarm(golden, client, config)
        self._log_distribution("train", train)
        self._log_distribution("eval", evals)

        train_path = out_dir / "sft.jsonl"

        # Stream rows to disk via a generator so we never materialise the
        # entire dataset in memory just to write it out.
        n_train = write_jsonl(
            train_path,
            (render(t, config.output_format) for t in train),
        )

        # Route the eval split through EvalPipeline so the procedure-level
        # leakage check actually runs in production (not just unit tests).
        # ``strict=False`` keeps existing call sites green: leakage emits a
        # warning rather than aborting the SFT run.
        eval_pipeline = EvalPipeline(output_format=config.output_format)
        eval_path = eval_pipeline.write(evals, train, out_dir, strict=config.eval_strict)

        logger.info(
            "wrote %d train rows to %s and %d eval rows to %s",
            n_train,
            train_path,
            len(evals),
            eval_path,
        )
        return train_path, eval_path

    @staticmethod
    def _log_distribution(label: str, trajectories: list[Trajectory]) -> None:
        if not trajectories:
            logger.info("%s split is empty", label)
            return
        by_difficulty: Counter[str] = Counter()
        by_procedure: Counter[str] = Counter()
        for t in trajectories:
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


__all__ = ["SFTPipeline"]
