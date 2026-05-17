"""Autonomous iteration loop.

`AutonomousRunner` runs the full pipeline, grades every axis with the
deterministic metrics in `gyroscope.quality`, and iterates the failing
phases until every axis crosses the configured threshold (default 95)
or the iteration budget is exhausted.

The mapping from failing axis → phase(s) to re-run:

    coverage          -> curation  (broaden golden doc)
    faithfulness      -> curation  (re-extract with stricter citation)
    diversity         -> sft       (regenerate trajectories, bump temperature)
    trainability      -> sft       (regenerate, enforce stricter critic_min_score)
    reward_soundness  -> rewards   (regenerate spec bundle)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from gyroscope.core.config import GyroscopeConfig
from gyroscope.core.io import write_jsonl
from gyroscope.core.llm import LLMClient
from gyroscope.core.models import Document, GoldenDocument, RewardSpec, Trajectory
from gyroscope.quality.metrics import QualityReport, assemble_report
from gyroscope.quality.report import render_report_html, render_report_markdown

logger = logging.getLogger(__name__)


AXIS_TO_PHASES: dict[str, list[str]] = {
    "coverage": ["curation"],
    "faithfulness": ["curation"],
    "diversity": ["sft"],
    "trainability": ["sft"],
    "reward_soundness": ["rewards"],
}


@dataclass
class IterationResult:
    iteration: int
    report: QualityReport
    phases_re_run: list[str] = field(default_factory=list)


@dataclass
class RunArtefacts:
    documents: list[Document]
    golden: GoldenDocument
    train: list[Trajectory]
    eval: list[Trajectory]
    rewards: list[RewardSpec]


class AutonomousRunner:
    """Drives an end-to-end run + iteration loop using the quality metrics."""

    def __init__(
        self, config: GyroscopeConfig, *, threshold: float = 95.0, max_iterations: int = 5
    ):
        self.config = config
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.history: list[IterationResult] = []

    # ----- phase callers (thin wrappers so tests can monkey-patch) -----

    async def _phase_ingest(self, client: LLMClient) -> list[Document]:
        from gyroscope.ingestion.pipeline import IngestionPipeline

        pipe = IngestionPipeline()
        docs = await pipe.ingest([str(p) for p in self.config.input_paths])
        out = self.config.output_dir / "documents.jsonl"
        write_jsonl(out, (d.model_dump(mode="json") for d in docs))
        return docs

    async def _phase_curate(self, documents: list[Document], client: LLMClient) -> GoldenDocument:
        from gyroscope.curation.pipeline import CurationPipeline

        golden = await CurationPipeline(self.config).distill(documents, client)
        (self.config.output_dir / "golden.md").write_text(golden.to_markdown(), encoding="utf-8")
        (self.config.output_dir / "golden.json").write_text(
            golden.model_dump_json(indent=2), encoding="utf-8"
        )
        return golden

    async def _phase_sft(
        self, golden: GoldenDocument, client: LLMClient
    ) -> tuple[list[Trajectory], list[Trajectory]]:
        from gyroscope.eval.pipeline import EvalPipeline
        from gyroscope.sft.formats import render
        from gyroscope.sft.swarm import run_swarm

        train, eval_ = await run_swarm(golden, client, self.config.sft)
        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.config.sft.output_format
        # SFT split: stream rows directly. Eval split: route through
        # EvalPipeline so the procedure-level leakage check runs in production
        # (strict=False so leakage warns rather than aborts, matching SFT
        # pipeline semantics and keeping iteration-loop tests green).
        write_jsonl(out_dir / "sft.jsonl", (render(t, fmt) for t in train))
        EvalPipeline(output_format=fmt).write(eval_, train, out_dir, strict=False)
        return train, eval_

    async def _phase_rewards(self, golden: GoldenDocument, client: LLMClient) -> list[RewardSpec]:
        from gyroscope.rewards.pipeline import RewardsPipeline

        bundle_path = await RewardsPipeline().run(
            golden, self.config.output_dir, client, self.config.rewards
        )
        # Read the bundle back to return specs.
        from gyroscope.rewards.spec import RewardBundle

        spec_path = bundle_path.parent / "reward_spec.yaml"
        bundle = RewardBundle.from_yaml(spec_path)
        return bundle.specs

    # ----- iteration loop -----

    def _grade(self, art: RunArtefacts) -> QualityReport:
        return assemble_report(
            golden=art.golden,
            source_texts=[d.text for d in art.documents],
            trajectories=art.train,
            reward_specs=art.rewards,
        )

    def _phases_to_rerun(self, report: QualityReport) -> list[str]:
        failing_phases: list[str] = []
        for axis_name, axis in report.axes.items():
            if axis.passed(self.threshold):
                continue
            for phase in AXIS_TO_PHASES.get(axis_name, []):
                if phase not in failing_phases:
                    failing_phases.append(phase)
        return failing_phases

    def _bump_config_for_retry(self, phase: str) -> None:
        """Mutate config in-place to make the next attempt produce different output."""
        if phase == "curation":
            # Loosen caps to capture more material.
            self.config.curation.max_principles = min(
                120, int(self.config.curation.max_principles * 1.5) + 1
            )
            self.config.curation.max_procedures = min(
                80, int(self.config.curation.max_procedures * 1.5) + 1
            )
            self.config.curation.max_knowledge_items = min(
                800, int(self.config.curation.max_knowledge_items * 1.5) + 1
            )
        if phase == "sft":
            # Bump diversity: more personas, higher temperature, stricter critic.
            self.config.sft.n_personas = min(24, self.config.sft.n_personas + 4)
            self.config.llm.temperature_swarm = min(1.0, self.config.llm.temperature_swarm + 0.1)
            self.config.sft.critic_min_score = min(0.9, self.config.sft.critic_min_score + 0.05)
        if phase == "rewards":
            self.config.rewards.reward_budget = max(4, self.config.rewards.reward_budget + 2)

    async def run(self) -> RunArtefacts:
        """Run end-to-end, iterating failing phases until all axes pass or budget hits."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        async with LLMClient(self.config) as client:
            documents = await self._phase_ingest(client)
            golden = await self._phase_curate(documents, client)
            train, eval_ = await self._phase_sft(golden, client)
            rewards = await self._phase_rewards(golden, client)

            art = RunArtefacts(
                documents=documents, golden=golden, train=train, eval=eval_, rewards=rewards
            )
            report = self._grade(art)
            self.history.append(IterationResult(iteration=0, report=report))
            logger.info("Iteration 0 overall: %.1f", report.overall())

            for it in range(1, self.max_iterations + 1):
                if report.all_pass(self.threshold):
                    logger.info("All axes >= %.1f after %d iteration(s).", self.threshold, it - 1)
                    break
                phases = self._phases_to_rerun(report)
                logger.info("Iteration %d: re-running phases %s", it, phases)
                for ph in phases:
                    self._bump_config_for_retry(ph)
                if "curation" in phases:
                    art.golden = await self._phase_curate(documents, client)
                if "sft" in phases or "curation" in phases:
                    art.train, art.eval = await self._phase_sft(art.golden, client)
                if "rewards" in phases or "curation" in phases:
                    art.rewards = await self._phase_rewards(art.golden, client)
                report = self._grade(art)
                self.history.append(
                    IterationResult(iteration=it, report=report, phases_re_run=phases)
                )
                logger.info("Iteration %d overall: %.1f", it, report.overall())

        self._write_report(report)
        return art

    def _write_report(self, report: QualityReport) -> None:
        run = self.config.output_dir
        run.mkdir(parents=True, exist_ok=True)
        (run / "report.md").write_text(render_report_markdown(report), encoding="utf-8")
        (run / "report.html").write_text(render_report_html(report), encoding="utf-8")
        (run / "report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        history = [
            {
                "iteration": h.iteration,
                "overall": h.report.overall(),
                "phases_re_run": h.phases_re_run,
                "axes": {k: v.score for k, v in h.report.axes.items()},
            }
            for h in self.history
        ]
        (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
