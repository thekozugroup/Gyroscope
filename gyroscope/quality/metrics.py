"""Deterministic, no-LLM-required quality metrics for pipeline artefacts.

These metrics are used in three places:
1. The CLI emits a final QualityReport.
2. The critique loop uses the per-axis scores to decide whether to re-run a phase.
3. Unit tests assert these scores degrade/improve predictably.

All metrics return 0..100. The intent is that a healthy pipeline lands all axes
at 95+; the autonomous loop iterates until everything is ≥ the target.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from gyroscope.core.models import (
    GoldenDocument,
    RewardSpec,
    Trajectory,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AxisScore:
    name: str
    score: float  # 0..100
    notes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def passed(self, threshold: float) -> bool:
        return self.score >= threshold


@dataclass
class QualityReport:
    """Summary of all axes across the pipeline."""

    axes: dict[str, AxisScore] = field(default_factory=dict)

    def add(self, axis: AxisScore) -> None:
        self.axes[axis.name] = axis

    def overall(self) -> float:
        if not self.axes:
            return 0.0
        return sum(a.score for a in self.axes.values()) / len(self.axes)

    def min_axis(self) -> AxisScore | None:
        if not self.axes:
            return None
        return min(self.axes.values(), key=lambda a: a.score)

    def all_pass(self, threshold: float) -> bool:
        return all(a.passed(threshold) for a in self.axes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall(), 2),
            "axes": {
                k: {"score": round(v.score, 2), "notes": v.notes, "details": v.details}
                for k, v in self.axes.items()
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _bounded(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Coverage — does the golden doc represent the BoK breadth?
# ---------------------------------------------------------------------------


def coverage_score(
    golden: GoldenDocument,
    source_texts: list[str],
    *,
    target_principles: int = 30,
    target_procedures: int = 15,
    target_knowledge: int = 100,
) -> AxisScore:
    """Coverage = breadth of golden vs. source AND vs. configured targets."""
    notes: list[str] = []

    # Density vs targets.
    p_ratio = min(1.0, len(golden.principles) / max(1, target_principles))
    pr_ratio = min(1.0, len(golden.procedures) / max(1, target_procedures))
    k_ratio = min(1.0, len(golden.knowledge) / max(1, target_knowledge))
    density = (p_ratio + pr_ratio + k_ratio) / 3.0
    notes.append(
        f"density vs targets: principles {len(golden.principles)}/{target_principles}, "
        f"procedures {len(golden.procedures)}/{target_procedures}, "
        f"knowledge {len(golden.knowledge)}/{target_knowledge}"
    )

    # Vocabulary coverage of source.
    source_vocab: set[str] = set()
    for txt in source_texts:
        source_vocab |= _token_set(txt)
    golden_text = golden.to_markdown()
    golden_vocab = _token_set(golden_text)
    if source_vocab:
        vocab_overlap = len(golden_vocab & source_vocab) / max(1, len(source_vocab))
    else:
        vocab_overlap = 0.0
    # Saturating overlap — 30% of source vocab in golden is already strong.
    vocab_signal = min(1.0, vocab_overlap / 0.30)
    notes.append(f"source-vocab overlap: {vocab_overlap:.2%}")

    score = _bounded(100 * (0.6 * density + 0.4 * vocab_signal))
    return AxisScore(
        name="coverage",
        score=score,
        notes=notes,
        details={
            "principles": len(golden.principles),
            "procedures": len(golden.procedures),
            "knowledge_items": len(golden.knowledge),
            "vocab_overlap": vocab_overlap,
        },
    )


# ---------------------------------------------------------------------------
# Faithfulness — does the golden / trajectories trace back to chunks?
# ---------------------------------------------------------------------------


def faithfulness_score(
    golden: GoldenDocument,
    trajectories: list[Trajectory] | None = None,
) -> AxisScore:
    """Penalise principles/knowledge items lacking source chunk citations.

    For trajectories: penalise any final assistant turn whose token-set has
    near-zero overlap with the golden document (likely hallucination).
    """
    notes: list[str] = []

    cited_principles = sum(1 for p in golden.principles if p.source_chunk_ids)
    cited_knowledge = sum(1 for k in golden.knowledge if k.citations)
    n_p = max(1, len(golden.principles))
    n_k = max(1, len(golden.knowledge))
    citation_rate = (cited_principles / n_p + cited_knowledge / n_k) / 2.0
    notes.append(
        f"citation rate: principles {cited_principles}/{n_p}, knowledge {cited_knowledge}/{n_k}"
    )

    trajectory_signal = 1.0
    if trajectories:
        golden_tokens = _token_set(golden.to_markdown())
        weak = 0
        for t in trajectories:
            assistant_last = next(
                (m.content for m in reversed(t.messages) if m.role == "assistant"),
                "",
            )
            if not assistant_last:
                continue
            overlap = _jaccard(_token_set(assistant_last), golden_tokens)
            if overlap < 0.05:
                weak += 1
        trajectory_signal = 1.0 - (weak / max(1, len(trajectories)))
        notes.append(f"trajectories with weak golden overlap: {weak}/{len(trajectories)}")

    score = _bounded(100 * (0.6 * citation_rate + 0.4 * trajectory_signal))
    return AxisScore(
        name="faithfulness",
        score=score,
        notes=notes,
        details={"citation_rate": citation_rate, "trajectory_signal": trajectory_signal},
    )


# ---------------------------------------------------------------------------
# Diversity — no degenerate repetition across the dataset.
# ---------------------------------------------------------------------------


def diversity_score(trajectories: list[Trajectory]) -> AxisScore:
    """Composite of (a) unique-first-user-message ratio, (b) entropy of
    tagged procedure ids, (c) inverse top-1 n-gram dominance across assistant
    final turns.
    """
    notes: list[str] = []
    if not trajectories:
        return AxisScore(name="diversity", score=0.0, notes=["no trajectories"])

    first_users: list[str] = []
    procedure_ids: list[str] = []
    assistant_finals: list[str] = []

    for t in trajectories:
        first_user = next((m.content for m in t.messages if m.role == "user"), "")
        first_users.append(first_user.strip().lower())
        pid_list = t.tags.get("procedure_ids") if t.tags else None
        if pid_list:
            procedure_ids.extend(pid_list)
        last_assistant = next(
            (m.content for m in reversed(t.messages) if m.role == "assistant"), ""
        )
        assistant_finals.append(last_assistant)

    unique_first = len(set(first_users)) / len(first_users)

    # Tag entropy.
    if procedure_ids:
        counts = Counter(procedure_ids)
        total = sum(counts.values())
        ps = [c / total for c in counts.values()]
        entropy = -sum(p * math.log(p) for p in ps if p > 0)
        max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
        tag_entropy = entropy / max_entropy
    else:
        tag_entropy = 0.5  # neutral if no tags

    # N-gram (4-gram) dominance.
    grams: Counter[tuple[str, ...]] = Counter()
    total_grams = 0
    for txt in assistant_finals:
        toks = _tokens(txt)
        for i in range(len(toks) - 3):
            grams[tuple(toks[i : i + 4])] += 1
            total_grams += 1
    if total_grams:
        top = grams.most_common(1)[0][1]
        dominance = top / total_grams
    else:
        dominance = 0.0
    ngram_signal = max(0.0, 1.0 - dominance * 20)  # 5% dominance halves the score

    notes.append(f"unique first-user msgs: {unique_first:.2%}")
    notes.append(f"procedure-tag entropy: {tag_entropy:.2f}")
    notes.append(f"top 4-gram dominance: {dominance:.2%}")

    score = _bounded(100 * (0.4 * unique_first + 0.3 * tag_entropy + 0.3 * ngram_signal))
    return AxisScore(
        name="diversity",
        score=score,
        notes=notes,
        details={
            "unique_first_users": unique_first,
            "tag_entropy": tag_entropy,
            "ngram_dominance": dominance,
        },
    )


# ---------------------------------------------------------------------------
# Trainability — format / schema / budget correctness.
# ---------------------------------------------------------------------------


def trainability_score(
    trajectories: list[Trajectory],
    *,
    max_tokens_per_example: int = 8000,
    min_messages: int = 2,
) -> AxisScore:
    """A trajectory is trainable iff it has system, ≥1 user/assistant turn pair,
    no empty content, alternating roles after system, fits the token budget."""
    notes: list[str] = []
    if not trajectories:
        return AxisScore(name="trainability", score=0.0, notes=["no trajectories"])

    n = len(trajectories)
    well_formed = 0
    over_budget = 0
    empty_turns = 0

    for t in trajectories:
        ok = True
        if not t.system.strip():
            ok = False
        if len(t.messages) < min_messages:
            ok = False
        prev = "system"
        for m in t.messages:
            if not m.content.strip():
                empty_turns += 1
                ok = False
            if m.role == prev and m.role in {"user", "assistant"}:
                ok = False
            prev = m.role
        approx_tokens = (len(t.system) + sum(len(m.content) for m in t.messages)) // 4
        if approx_tokens > max_tokens_per_example:
            over_budget += 1
            ok = False
        if ok:
            well_formed += 1

    rate = well_formed / n
    notes.append(f"well-formed: {well_formed}/{n}")
    notes.append(f"over-budget: {over_budget}/{n}")
    notes.append(f"empty turns: {empty_turns}")

    score = _bounded(100 * rate)
    return AxisScore(
        name="trainability",
        score=score,
        notes=notes,
        details={
            "well_formed": well_formed,
            "over_budget": over_budget,
            "empty_turns": empty_turns,
        },
    )


# ---------------------------------------------------------------------------
# Reward soundness — calibrated, non-degenerate, gameability-resistant.
# ---------------------------------------------------------------------------


def reward_soundness_score(
    specs: list[RewardSpec],
    *,
    golden: GoldenDocument | None = None,
) -> AxisScore:
    """Heuristics:
    - has at least one safety, one principle, one procedure reward
    - weights sum to a sensible total (not dominated by one reward)
    - no reward references missing principle/procedure ids
    - lexical/format rewards aren't trivially satisfied by an empty string
    """
    notes: list[str] = []
    if not specs:
        return AxisScore(name="reward_soundness", score=0.0, notes=["no rewards"])

    kinds = {s.kind.value if hasattr(s.kind, "value") else str(s.kind) for s in specs}
    must_have = {"safety", "principle", "procedure"}
    coverage = len(kinds & must_have) / len(must_have)
    notes.append(f"kind coverage: {sorted(kinds & must_have)} (need {sorted(must_have)})")

    # No reward dominates total weight by more than 50%.
    total_w = sum(s.weight for s in specs) or 1.0
    max_share = max(s.weight for s in specs) / total_w
    weight_signal = 1.0 if max_share <= 0.5 else max(0.0, 1.0 - (max_share - 0.5) * 2)
    notes.append(f"max weight share: {max_share:.2%}")

    # Reference integrity.
    ref_ok = 1.0
    if golden:
        valid_pids = {p.id for p in golden.principles}
        valid_prids = {p.id for p in golden.procedures}
        broken = 0
        total_refs = 0
        for s in specs:
            for pid in s.principle_ids:
                total_refs += 1
                if pid not in valid_pids:
                    broken += 1
            for prid in s.procedure_ids:
                total_refs += 1
                if prid not in valid_prids:
                    broken += 1
        if total_refs:
            ref_ok = 1.0 - broken / total_refs
            notes.append(f"broken refs: {broken}/{total_refs}")

    # Trivial-satisfaction probe.
    trivial = 0
    for s in specs:
        kind = s.kind.value if hasattr(s.kind, "value") else str(s.kind)
        if kind == "lexical" and not s.config.get("required"):
            trivial += 1
        if kind == "format" and not (
            s.config.get("pattern") or s.config.get("json_schema") or s.config.get("sections")
        ):
            trivial += 1
    trivial_signal = 1.0 - trivial / max(1, len(specs))
    notes.append(f"trivially satisfiable rewards: {trivial}/{len(specs)}")

    score = _bounded(
        100 * (0.35 * coverage + 0.25 * weight_signal + 0.25 * ref_ok + 0.15 * trivial_signal)
    )
    return AxisScore(
        name="reward_soundness",
        score=score,
        notes=notes,
        details={
            "coverage": coverage,
            "max_weight_share": max_share,
            "ref_integrity": ref_ok,
            "trivial_count": trivial,
        },
    )


# ---------------------------------------------------------------------------
# Convenience: assemble all axes for a full pipeline run.
# ---------------------------------------------------------------------------


def assemble_report(
    *,
    golden: GoldenDocument,
    source_texts: list[str],
    trajectories: list[Trajectory],
    reward_specs: list[RewardSpec],
) -> QualityReport:
    report = QualityReport()
    report.add(coverage_score(golden, source_texts))
    report.add(faithfulness_score(golden, trajectories))
    report.add(diversity_score(trajectories))
    report.add(trainability_score(trajectories))
    report.add(reward_soundness_score(reward_specs, golden=golden))
    return report


__all__ = [
    "AxisScore",
    "QualityReport",
    "assemble_report",
    "coverage_score",
    "diversity_score",
    "faithfulness_score",
    "reward_soundness_score",
    "trainability_score",
]
