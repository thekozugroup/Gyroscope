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
from typing import Any

from gyroscope.core.config import GyroscopeConfig
from gyroscope.core.io import write_jsonl
from gyroscope.core.llm import LLMClient
from gyroscope.core.models import Document, GoldenDocument, RewardSpec, Trajectory
from gyroscope.quality.metrics import QualityReport, assemble_report
from gyroscope.quality.report import write_report_artefacts

logger = logging.getLogger(__name__)


AXIS_TO_PHASES: dict[str, list[str]] = {
    "coverage": ["curation"],
    "faithfulness": ["curation"],
    "diversity": ["sft"],
    "trainability": ["sft"],
    "reward_soundness": ["rewards"],
}


def register_axis_remediation(axis: str, phases: list[str]) -> None:
    """Register the phases the autonomous runner should re-run when ``axis`` fails.

    Plugins that add a new quality metric in :mod:`gyroscope.quality.metrics`
    use this to declare which pipeline phase to re-run on failure without
    monkey-patching the module-level :data:`AXIS_TO_PHASES`.
    """
    if not phases:
        raise ValueError("phases must be a non-empty list")
    AXIS_TO_PHASES[axis] = list(phases)


def unregister_axis_remediation(axis: str) -> bool:
    """Remove a previously-registered axis remediation. Returns True on hit.

    Useful for test teardown so registered fakes do not leak between tests.
    """
    return AXIS_TO_PHASES.pop(axis, None) is not None


def axis_remediation_map() -> dict[str, tuple[str, ...]]:
    """Read-only snapshot of the axis -> phases mapping."""
    return {axis: tuple(phases) for axis, phases in AXIS_TO_PHASES.items()}


@dataclass
class IterationResult:
    iteration: int
    report: QualityReport
    phases_re_run: list[str] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    """Snapshot of the mutable knobs that produced this iteration's artefacts."""


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
        # CurationPipeline.distill is the single writer for golden.{md,json};
        # see CurationPipeline._write_outputs. We do not duplicate the write
        # here so the two paths cannot drift in serialisation format.
        from gyroscope.curation.pipeline import CurationPipeline

        return await CurationPipeline(self.config).distill(documents, client)

    async def _phase_sft(
        self, golden: GoldenDocument, client: LLMClient
    ) -> tuple[list[Trajectory], list[Trajectory]]:
        import json

        from gyroscope.eval.pipeline import EvalPipeline
        from gyroscope.sft.formats import render
        from gyroscope.sft.swarm import stream_swarm

        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.config.sft.output_format
        train_path = out_dir / "sft.jsonl"

        # Stream the train split straight to disk as each survivor is yielded.
        # Keep only stripped-down metadata copies for the leakage check (tags
        # + scenario_id; messages zeroed) and the in-memory train return list
        # — callers (the quality grader) need procedure_ids + at least one
        # message to score diversity/faithfulness.
        train_full: list[Trajectory] = []
        evals: list[Trajectory] = []
        with train_path.open("w", encoding="utf-8") as fh:
            async for traj, split in stream_swarm(golden, client, self.config.sft):
                if split == "train":
                    fh.write(json.dumps(render(traj, fmt), ensure_ascii=False))
                    fh.write("\n")
                    train_full.append(traj)
                else:
                    evals.append(traj)

        EvalPipeline(output_format=fmt).write(
            evals, train_full, out_dir, strict=self.config.sft.eval_strict
        )
        return train_full, evals

    async def _phase_rewards(self, golden: GoldenDocument, client: LLMClient) -> list[RewardSpec]:
        from gyroscope.rewards.pipeline import RewardsPipeline

        bundle_path = await RewardsPipeline().run(
            golden, self.config.output_dir, client, config=self.config.rewards
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
        """Mutate config in-place to make the next attempt produce different output.

        Ceilings here are *soft* — a higher user-supplied baseline is never
        reduced. Each bump is the larger of the current value and the
        ``int(current * 1.5) + 1`` growth target, capped at the soft ceiling
        only when the current value is at or below it.
        """

        def grow(current: int, growth_cap: int) -> int:
            target = int(current * 1.5) + 1
            return max(current, min(growth_cap, target))

        if phase == "curation":
            self.config.curation.max_principles = grow(self.config.curation.max_principles, 120)
            self.config.curation.max_procedures = grow(self.config.curation.max_procedures, 80)
            self.config.curation.max_knowledge_items = grow(
                self.config.curation.max_knowledge_items, 800
            )
        if phase == "sft":
            # Bump diversity: more personas, higher temperature, stricter critic.
            self.config.sft.n_personas = max(
                self.config.sft.n_personas, self.config.sft.n_personas + 4
            )
            self.config.llm.temperature_swarm = min(1.0, self.config.llm.temperature_swarm + 0.1)
            # Soft ceiling on critic_min_score so a user who set 0.95 is not
            # silently lowered to 0.9 on the first retry.
            self.config.sft.critic_min_score = max(
                self.config.sft.critic_min_score,
                min(0.9, self.config.sft.critic_min_score + 0.05),
            )
        if phase == "rewards":
            self.config.rewards.reward_budget = max(
                self.config.rewards.reward_budget, self.config.rewards.reward_budget + 2
            )

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
            self.history.append(
                IterationResult(iteration=0, report=report, config_snapshot=self._config_snapshot())
            )
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
                next_report = self._grade(art)
                # No-progress guard: if the overall score did not improve AND
                # the same phases failed, the loop is structurally stuck (e.g.
                # the golden doc just cannot produce a reward bundle because
                # the include_kinds list is empty). Bail with a warning rather
                # than burn the remaining iteration budget.
                if (
                    next_report.overall() <= report.overall() + 0.5
                    and self._phases_to_rerun(next_report) == phases
                ):
                    logger.warning(
                        "Iteration %d made no progress (overall %.1f -> %.1f) on "
                        "the same failing phases %s — aborting retry loop. "
                        "Consider widening the inputs or relaxing the threshold.",
                        it,
                        report.overall(),
                        next_report.overall(),
                        phases,
                    )
                    self.history.append(
                        IterationResult(
                            iteration=it,
                            report=next_report,
                            phases_re_run=phases,
                            config_snapshot=self._config_snapshot(),
                        )
                    )
                    report = next_report
                    break
                report = next_report
                self.history.append(
                    IterationResult(
                        iteration=it,
                        report=report,
                        phases_re_run=phases,
                        config_snapshot=self._config_snapshot(),
                    )
                )
                logger.info("Iteration %d overall: %.1f", it, report.overall())

        self._write_report(report)
        return art

    def _config_snapshot(self) -> dict[str, Any]:
        """Snapshot the knobs the retry loop mutates so each iteration's
        artefacts can be traced back to the exact config that produced them."""
        c = self.config
        return {
            "curation": {
                "max_principles": c.curation.max_principles,
                "max_procedures": c.curation.max_procedures,
                "max_knowledge_items": c.curation.max_knowledge_items,
                "dedup_threshold": c.curation.dedup_threshold,
                "synthesizer_dedup_threshold": c.curation.synthesizer_dedup_threshold,
            },
            "sft": {
                "n_personas": c.sft.n_personas,
                "critic_min_score": c.sft.critic_min_score,
                "n_trajectories": c.sft.n_trajectories,
            },
            "rewards": {"reward_budget": c.rewards.reward_budget},
            "llm": {
                "temperature_swarm": c.llm.temperature_swarm,
                "temperature_critic": c.llm.temperature_critic,
            },
        }

    def _write_report(self, report: QualityReport) -> None:
        run = self.config.output_dir
        write_report_artefacts(report, run)
        history = [
            {
                "iteration": h.iteration,
                "overall": h.report.overall(),
                "phases_re_run": h.phases_re_run,
                "axes": {k: v.score for k, v in h.report.axes.items()},
                "config": h.config_snapshot,
            }
            for h in self.history
        ]
        (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
