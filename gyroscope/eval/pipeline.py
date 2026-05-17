"""Eval set writer + leakage check.

The actual trajectory generation lives in `gyroscope.sft.swarm.run_swarm`,
which returns both a training set and a held-out eval set. This module is
the thin layer that persists the eval set and verifies it does not share
procedure ids with the training set.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gyroscope.core.io import write_jsonl
from gyroscope.core.models import Trajectory

logger = logging.getLogger(__name__)


def _procedure_ids(trajectories: list[Trajectory]) -> set[str]:
    ids: set[str] = set()
    for t in trajectories:
        pids = t.tags.get("procedure_ids") if t.tags else None
        if isinstance(pids, list):
            ids.update(pids)
    return ids


def leakage_check(train: list[Trajectory], eval_: list[Trajectory]) -> set[str]:
    """Return the set of procedure ids that appear in both splits.

    Empty set ⇒ no procedure-level leakage. Callers should raise or warn
    when this is non-empty depending on strictness.
    """
    return _procedure_ids(train) & _procedure_ids(eval_)


class EvalPipeline:
    """Writes eval JSONL and verifies leakage. Format mirrors the SFT writer.

    The constructor and writer both read :data:`gyroscope.sft.formats.
    FORMAT_WRITERS` so any format registered via :func:`gyroscope.sft.
    formats.register_format` works here without code changes.
    """

    def __init__(self, output_format: str = "sharegpt") -> None:
        # Lazy import to avoid a hard dep on sft.formats during partial installs.
        from gyroscope.sft.formats import FORMAT_WRITERS

        if output_format not in FORMAT_WRITERS:
            raise ValueError(
                f"Unsupported format: {output_format!r}. "
                f"Registered formats: {sorted(FORMAT_WRITERS)}"
            )
        self.output_format = output_format

    def write(
        self,
        eval_trajectories: list[Trajectory],
        train_trajectories: list[Trajectory],
        output_dir: Path | str,
        *,
        strict: bool = True,
    ) -> Path:
        # Lazy import keeps this consistent with the constructor and avoids
        # pulling sft.formats at import time.
        from gyroscope.sft.formats import FORMAT_WRITERS

        writer = FORMAT_WRITERS[self.output_format]

        leaked = leakage_check(train_trajectories, eval_trajectories)
        if leaked:
            msg = (
                f"Procedure leakage between train and eval: {sorted(leaked)}. "
                "Eval procedures must be disjoint from train procedures."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_path = out_dir / "eval.jsonl"
        n = write_jsonl(eval_path, (writer(t) for t in eval_trajectories))
        logger.info("Wrote %d eval trajectories to %s", n, eval_path)
        return eval_path
