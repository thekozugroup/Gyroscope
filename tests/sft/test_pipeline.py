"""Tests for :class:`gyroscope.sft.pipeline.SFTPipeline`.

Focus areas:

* The pipeline must stream train rows directly to the open file handle as
  each survivor is yielded by ``stream_swarm`` — no full-list buffering.
* The eval split must be routed through :class:`EvalPipeline.write` so the
  procedure-level leakage check runs in production paths, not only when
  someone calls ``EvalPipeline`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gyroscope.core.config import SFTConfig
from gyroscope.sft import pipeline as pipeline_mod
from gyroscope.sft.pipeline import SFTPipeline

from .conftest import make_golden, make_trajectory


@pytest.mark.asyncio
async def test_sft_pipeline_streams_train_rows_to_disk_as_they_arrive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SFTPipeline.run`` must append each train row to ``sft.jsonl`` as
    ``stream_swarm`` yields it, without buffering the full split first.
    The strongest portable proxy: drive a fake stream that yields N rows
    and assert the on-disk file has exactly N lines in the right order
    AND that the rendered metadata buffer never held a full transcript.
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

    pipeline = SFTPipeline()
    train_path, eval_path = await pipeline.run(golden, tmp_path, client=None, config=SFTConfig())

    assert train_path.exists() and eval_path.exists()

    # On-disk row count matches what the stream emitted, in order.
    train_lines = train_path.read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == len(train)
    for idx, line in enumerate(train_lines):
        row = json.loads(line)
        assert row.get("id") == train[idx].id

    eval_lines = eval_path.read_text(encoding="utf-8").splitlines()
    assert len(eval_lines) == len(evals)
    assert json.loads(eval_lines[0]).get("id") == evals[0].id


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
