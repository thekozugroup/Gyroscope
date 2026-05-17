"""Gyroscope CLI.

Single entry point: `gyroscope`. Subcommands map to phases so any stage can
be re-run from a previous artefact.

    gyroscope ingest   --input ./bok --output ./runs/foo
    gyroscope curate   --run ./runs/foo
    gyroscope sft      --run ./runs/foo
    gyroscope rewards  --run ./runs/foo
    gyroscope run      --input ./bok --output ./runs/foo   # end-to-end
    gyroscope report   --run ./runs/foo                    # quality report only
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from gyroscope.core.config import GyroscopeConfig
from gyroscope.core.io import read_jsonl, write_jsonl, write_yaml
from gyroscope.core.logging import setup_logging
from gyroscope.core.models import Document, GoldenDocument, RewardSpec, Trajectory

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
console = Console()
log = logging.getLogger("gyroscope.cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(run_dir: Path) -> GyroscopeConfig:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return GyroscopeConfig.model_validate_json(cfg_path.read_text())
    return GyroscopeConfig(output_dir=run_dir)


def _save_config(cfg: GyroscopeConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2, exclude={"api_key"}))


def _write_documents(docs: list[Document], path: Path) -> None:
    write_jsonl(path, (d.model_dump(mode="json") for d in docs))


def _read_documents(path: Path) -> list[Document]:
    return [Document.model_validate(row) for row in read_jsonl(path)]


def _write_golden(g: GoldenDocument, run_dir: Path) -> None:
    (run_dir / "golden.md").write_text(g.to_markdown(), encoding="utf-8")
    (run_dir / "golden.json").write_text(g.model_dump_json(indent=2), encoding="utf-8")


def _read_golden(run_dir: Path) -> GoldenDocument:
    return GoldenDocument.model_validate_json((run_dir / "golden.json").read_text())


def _read_trajectories_from_sharegpt(path: Path) -> list[Trajectory]:
    # Lazy import to avoid circular when phases aren't installed.
    from gyroscope.sft.formats import from_sharegpt

    return [from_sharegpt(row) for row in read_jsonl(path)]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    input_: Annotated[list[Path], typer.Option("--input", "-i", help="Path(s) or URL(s).")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """Ingest BoK sources into a normalised document JSONL."""
    setup_logging(log_level)
    from gyroscope.ingestion.pipeline import IngestionPipeline

    cfg = _load_config(output)
    cfg.input_paths = list(input_)
    cfg.output_dir = output
    _save_config(cfg, output)

    pipe = IngestionPipeline()
    docs = asyncio.run(pipe.ingest([str(p) for p in input_]))
    out_path = output / "documents.jsonl"
    _write_documents(docs, out_path)
    console.print(f"[green]Ingested[/green] {len(docs)} documents -> {out_path}")


@app.command()
def curate(
    run: Annotated[Path, typer.Option("--run", "-r")],
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """Distill ingested documents into the golden document."""
    setup_logging(log_level)
    from gyroscope.core.llm import LLMClient
    from gyroscope.curation.pipeline import CurationPipeline

    cfg = _load_config(run)
    docs = _read_documents(run / "documents.jsonl")

    async def _go() -> GoldenDocument:
        async with LLMClient(cfg) as client:
            return await CurationPipeline(cfg).distill(docs, client)

    golden = asyncio.run(_go())
    _write_golden(golden, run)
    console.print(f"[green]Golden document[/green] written to {run / 'golden.md'}")


@app.command()
def sft(
    run: Annotated[Path, typer.Option("--run", "-r")],
    n_trajectories: Annotated[Optional[int], typer.Option()] = None,
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """Generate the SFT dataset from the golden document."""
    setup_logging(log_level)
    from gyroscope.core.llm import LLMClient
    from gyroscope.sft.pipeline import SFTPipeline

    cfg = _load_config(run)
    if n_trajectories:
        cfg.sft.n_trajectories = n_trajectories
    golden = _read_golden(run)

    async def _go() -> tuple[Path, Path]:
        async with LLMClient(cfg) as client:
            return await SFTPipeline().run(golden, run, client, cfg.sft)

    sft_path, eval_path = asyncio.run(_go())
    console.print(f"[green]SFT[/green]   -> {sft_path}")
    console.print(f"[green]Eval[/green]  -> {eval_path}")


@app.command()
def rewards(
    run: Annotated[Path, typer.Option("--run", "-r")],
    budget: Annotated[Optional[int], typer.Option("--budget", "-b")] = None,
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """Design and emit reward functions from the golden document."""
    setup_logging(log_level)
    from gyroscope.core.llm import LLMClient
    from gyroscope.rewards.pipeline import RewardsPipeline

    cfg = _load_config(run)
    if budget:
        cfg.rewards.reward_budget = budget
    golden = _read_golden(run)

    async def _go() -> Path:
        async with LLMClient(cfg) as client:
            return await RewardsPipeline().run(golden, run, client, cfg.rewards)

    out = asyncio.run(_go())
    console.print(f"[green]Rewards[/green] -> {out}")


@app.command()
def report(
    run: Annotated[Path, typer.Option("--run", "-r")],
    threshold: Annotated[float, typer.Option()] = 95.0,
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """Compute and render the quality report for an existing run."""
    setup_logging(log_level)
    from gyroscope.quality.metrics import assemble_report
    from gyroscope.quality.report import render_report_html, render_report_markdown

    golden = _read_golden(run)
    docs = _read_documents(run / "documents.jsonl")
    trajectories = _read_trajectories_from_sharegpt(run / "sft.jsonl")
    reward_spec_path = run / "rewards" / "reward_spec.yaml"
    specs: list[RewardSpec] = []
    if reward_spec_path.exists():
        import yaml

        data = yaml.safe_load(reward_spec_path.read_text())
        specs = [RewardSpec.model_validate(s) for s in data.get("specs", [])]

    report = assemble_report(
        golden=golden,
        source_texts=[d.text for d in docs],
        trajectories=trajectories,
        reward_specs=specs,
    )

    md_path = run / "report.md"
    html_path = run / "report.html"
    json_path = run / "report.json"
    md_path.write_text(render_report_markdown(report), encoding="utf-8")
    html_path.write_text(render_report_html(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    table = Table(title=f"Quality Report — overall {report.overall():.1f} / 100")
    table.add_column("Axis")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    for name, axis in report.axes.items():
        status = "[green]PASS[/green]" if axis.passed(threshold) else "[red]FAIL[/red]"
        table.add_row(name, f"{axis.score:.1f}", status)
    console.print(table)

    if not report.all_pass(threshold):
        raise typer.Exit(code=1)


@app.command()
def run(
    input_: Annotated[list[Path], typer.Option("--input", "-i")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    n_trajectories: Annotated[int, typer.Option()] = 1000,
    reward_budget: Annotated[int, typer.Option()] = 12,
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """End-to-end: ingest → curate → sft → rewards → report."""
    ingest(input_=input_, output=output, log_level=log_level)
    curate(run=output, log_level=log_level)
    sft(run=output, n_trajectories=n_trajectories, log_level=log_level)
    rewards(run=output, budget=reward_budget, log_level=log_level)
    report(run=output, log_level=log_level)


if __name__ == "__main__":
    app()
