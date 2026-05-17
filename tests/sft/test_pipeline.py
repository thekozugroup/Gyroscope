"""Tests for :class:`gyroscope.sft.pipeline.SFTPipeline`.

Focus areas:

* The pipeline must stream rows to ``write_jsonl`` via a generator — not a
  list — so we never materialise the entire dataset in memory just to
  serialise it.
* The eval split must be routed through :class:`EvalPipeline.write` so the
  procedure-level leakage check runs in production paths, not only when
  someone calls ``EvalPipeline`` directly.
"""

from __future__ import annotations

from pathlib import Path
from types import GeneratorType
from typing import Any

import pytest

from gyroscope.core.config import SFTConfig
from gyroscope.sft import pipeline as pipeline_mod
from gyroscope.sft.pipeline import SFTPipeline

from .conftest import make_golden, make_trajectory


@pytest.mark.asyncio
async def test_sft_pipeline_streams_rows_via_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SFTPipeline.run`` must hand ``write_jsonl`` a generator (lazy
    iterator) for the train split, not a pre-built list.
    """
    golden = make_golden()
    train = [
        make_trajectory(
            id_=f"TRJ-{i:04d}",
            scenario_id=f"SCN-{i:04d}",
            tags={
                "procedure_ids": [f"PRC-{i:04d}"],
                "principle_ids": ["PRN-0001"],
                "persona": "PER-0001",
                "difficulty": "medium",
            },
        )
        for i in range(1, 4)
    ]
    evals = [
        make_trajectory(
            id_="TRJ-9999",
            scenario_id="SCN-9999",
            tags={
                "procedure_ids": ["PRC-9999"],
                "principle_ids": ["PRN-0001"],
                "persona": "PER-0001",
                "difficulty": "medium",
            },
        )
    ]

    async def fake_stream_swarm(_golden: Any, _client: Any, _config: Any):
        for t in train:
            yield t, "train"
        for t in evals:
            yield t, "eval"

    monkeypatch.setattr(pipeline_mod, "stream_swarm", fake_stream_swarm)

    captured: list[Any] = []
    real_write_jsonl = pipeline_mod.write_jsonl

    def spy_write_jsonl(path: Any, rows: Any) -> int:
        captured.append(rows)
        # Force the iterator so the file actually gets written for downstream
        # assertions (e.g. EvalPipeline.write inside the SFT pipeline calls
        # write_jsonl too; both invocations must be observed).
        rows_list = list(rows) if not isinstance(rows, list) else rows
        return real_write_jsonl(path, rows_list)

    monkeypatch.setattr(pipeline_mod, "write_jsonl", spy_write_jsonl)

    pipeline = SFTPipeline()
    train_path, eval_path = await pipeline.run(golden, tmp_path, client=None, config=SFTConfig())

    assert train_path.exists()
    assert eval_path.exists()
    # At least one call to write_jsonl came from SFTPipeline itself (the
    # train split) and it MUST be a generator, not a list.
    assert captured, "expected at least one write_jsonl call"
    train_rows_arg = captured[0]
    assert isinstance(train_rows_arg, GeneratorType), (
        f"SFTPipeline.run must stream train rows via generator, got {type(train_rows_arg).__name__}"
    )


@pytest.mark.asyncio
async def test_sft_pipeline_routes_eval_split_through_eval_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SFTPipeline.run`` must call ``EvalPipeline.write`` for the eval
    split so the leakage check runs in production, not only in unit tests.
    """
    golden = make_golden()
    train = [
        make_trajectory(
            id_="TRJ-0001",
            scenario_id="SCN-0001",
            tags={
                "procedure_ids": ["PRC-0001"],
                "principle_ids": ["PRN-0001"],
                "persona": "PER-0001",
                "difficulty": "medium",
            },
        )
    ]
    evals = [
        make_trajectory(
            id_="TRJ-9999",
            scenario_id="SCN-9999",
            tags={
                "procedure_ids": ["PRC-9999"],
                "principle_ids": ["PRN-0001"],
                "persona": "PER-0001",
                "difficulty": "medium",
            },
        )
    ]

    async def fake_stream_swarm(_golden: Any, _client: Any, _config: Any):
        for t in train:
            yield t, "train"
        for t in evals:
            yield t, "eval"

    monkeypatch.setattr(pipeline_mod, "stream_swarm", fake_stream_swarm)

    write_calls: list[dict[str, Any]] = []
    real_write = pipeline_mod.EvalPipeline.write

    def recording_write(self, eval_trajectories, train_trajectories, output_dir, *, strict=True):  # type: ignore[no-untyped-def]
        write_calls.append(
            {
                "self": self,
                "n_eval": len(eval_trajectories),
                "n_train": len(train_trajectories),
                "strict": strict,
            }
        )
        return real_write(
            self,
            eval_trajectories,
            train_trajectories,
            output_dir,
            strict=strict,
        )

    monkeypatch.setattr(pipeline_mod.EvalPipeline, "write", recording_write)

    pipeline = SFTPipeline()
    train_path, eval_path = await pipeline.run(golden, tmp_path, client=None, config=SFTConfig())

    assert train_path.exists() and eval_path.exists()
    assert len(write_calls) == 1, "EvalPipeline.write must be invoked exactly once"
    call = write_calls[0]
    assert call["n_eval"] == 1
    assert call["n_train"] == 1
    # strict=False so leakage emits a warning rather than blowing up the run.
    assert call["strict"] is False
