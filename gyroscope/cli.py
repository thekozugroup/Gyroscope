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
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from gyroscope.core.config import GyroscopeConfig
from gyroscope.core.io import read_jsonl, write_jsonl
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
    n_trajectories: Annotated[int | None, typer.Option()] = None,
    output_format: Annotated[
        str | None,
        typer.Option("--format", "-f", help="sharegpt | chatml | alpaca"),
    ] = None,
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """Generate the SFT dataset from the golden document."""
    setup_logging(log_level)
    from gyroscope.core.llm import LLMClient
    from gyroscope.sft.formats import FORMAT_WRITERS
    from gyroscope.sft.pipeline import SFTPipeline

    cfg = _load_config(run)
    if n_trajectories is not None:
        cfg.sft.n_trajectories = n_trajectories
    if output_format is not None:
        if output_format not in FORMAT_WRITERS:
            raise typer.BadParameter(
                f"Unknown format {output_format!r}; available: {sorted(FORMAT_WRITERS)}"
            )
        cfg.sft.output_format = output_format  # type: ignore[assignment]
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
    budget: Annotated[int | None, typer.Option("--budget", "-b")] = None,
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """Design and emit reward functions from the golden document."""
    setup_logging(log_level)
    from gyroscope.core.llm import LLMClient
    from gyroscope.rewards.pipeline import RewardsPipeline

    cfg = _load_config(run)
    if budget is not None:
        cfg.rewards.reward_budget = budget
    golden = _read_golden(run)

    async def _go() -> Path:
        async with LLMClient(cfg) as client:
            return await RewardsPipeline().run(golden, run, client, config=cfg.rewards)

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
    from gyroscope.quality.report import write_report_artefacts

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

    write_report_artefacts(report, run)

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
    threshold: Annotated[float, typer.Option(help="Quality threshold per axis.")] = 95.0,
    max_iterations: Annotated[
        int, typer.Option(help="Maximum autonomous-fix iterations after the initial run.")
    ] = 5,
    output_format: Annotated[
        str | None,
        typer.Option("--format", "-f", help="sharegpt | chatml | alpaca"),
    ] = None,
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """End-to-end run driven by :class:`AutonomousRunner`.

    Ingests the inputs, distills the golden document, generates the SFT
    dataset, designs the reward functions, and re-runs any failing phase
    until every quality axis crosses ``--threshold`` or ``--max-iterations``
    is exhausted. Reports land in ``<output>/report.{md,html,json}``.
    """
    setup_logging(log_level)
    from gyroscope.runner import AutonomousRunner
    from gyroscope.sft.formats import FORMAT_WRITERS

    cfg = GyroscopeConfig(input_paths=list(input_), output_dir=output)
    cfg.sft.n_trajectories = n_trajectories
    cfg.rewards.reward_budget = reward_budget
    if output_format is not None:
        if output_format not in FORMAT_WRITERS:
            raise typer.BadParameter(
                f"Unknown format {output_format!r}; available: {sorted(FORMAT_WRITERS)}"
            )
        cfg.sft.output_format = output_format  # type: ignore[assignment]
    cfg.log_level = log_level
    _save_config(cfg, output)

    runner = AutonomousRunner(cfg, threshold=threshold, max_iterations=max_iterations)
    asyncio.run(runner.run())

    final_report = runner.history[-1].report
    table = Table(title=f"Quality Report — overall {final_report.overall():.1f} / 100")
    table.add_column("Axis")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    for name, axis in final_report.axes.items():
        status = "[green]PASS[/green]" if axis.passed(threshold) else "[red]FAIL[/red]"
        table.add_row(name, f"{axis.score:.1f}", status)
    console.print(table)
    console.print(
        f"[bold]{len(runner.history)}[/bold] iteration(s); "
        f"history at [cyan]{output / 'history.json'}[/cyan]"
    )
    if not final_report.all_pass(threshold):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
