"""Deterministic designer that turns a GoldenDocument into a RewardBundle.

The designer is intentionally mostly LLM-free: it inspects the golden document
structurally and emits a balanced set of reward specs. The :class:`LLMClient`
argument is accepted (to match the phase-pipeline signature used elsewhere) but
not currently invoked — the design is reproducible from the golden document
alone, which matters for unit-testability and offline runs.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import OrderedDict
from typing import TYPE_CHECKING

from gyroscope.core.models import (
    AntiPattern,
    GoldenDocument,
    Principle,
    Procedure,
    RewardKind,
    RewardSpec,
    VocabularyTerm,
)
from gyroscope.rewards.spec import RewardBundle

if TYPE_CHECKING:
    from gyroscope.core.config import RewardConfig
    from gyroscope.core.llm import LLMClient

logger = logging.getLogger(__name__)

__all__ = ["design_rewards", "priority_for_kind"]


# Lower number = higher priority. Used to drop specs when over budget.
_PRIORITY: dict[RewardKind, int] = {
    RewardKind.SAFETY: 0,
    RewardKind.PROCEDURE: 1,
    RewardKind.PRINCIPLE: 2,
    RewardKind.CITATION: 3,
    RewardKind.FORMAT: 4,
    RewardKind.LEXICAL: 5,
    RewardKind.LENGTH: 6,
}


def priority_for_kind(kind: RewardKind) -> int:
    """Public accessor used in tests."""
    return _PRIORITY[kind]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NAME_SANITISE_RE = re.compile(r"[^a-z0-9_]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _to_identifier(prefix: str, raw: str) -> str:
    """Build a stable Python identifier from arbitrary text."""
    base = _NAME_SANITISE_RE.sub("_", raw.lower()).strip("_")
    base = base or "x"
    return f"{prefix}_{base}"[:64]


def _step_keywords(step_action: str, *, max_keywords: int = 3) -> list[str]:
    """Pick the first few non-stopword tokens from a step as keyword hooks."""
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "your",
        "you",
        "are",
        "use",
        "ensure",
        "make",
        "this",
        "that",
        "then",
        "must",
        "have",
        "any",
        "all",
        "step",
        "should",
        "when",
        "where",
        "while",
    }
    words = [w.lower() for w in _WORD_RE.findall(step_action)]
    out: list[str] = []
    for w in words:
        if w in stop or len(w) < 3:
            continue
        if w in out:
            continue
        out.append(w)
        if len(out) >= max_keywords:
            break
    return out or [w for w in words[:1]] or [step_action.strip()[:24].lower()]


def _avg_procedure_token_length(procedures: list[Procedure]) -> int:
    if not procedures:
        return 350
    lengths: list[int] = []
    for p in procedures:
        text = " ".join(s.action for s in p.steps) + " " + p.purpose
        lengths.append(max(1, len(text.split())))
    return max(80, int(statistics.mean(lengths)) * 3)


def _principles_asking_for_citations(principles: list[Principle]) -> bool:
    needles = ("cite", "citation", "source", "reference", "[knw")
    for p in principles:
        s = p.statement.lower()
        if any(n in s for n in needles):
            return True
    return False


def _principles_asking_for_structure(principles: list[Principle]) -> bool:
    needles = ("section", "structure", "headings", "headers", "markdown", "format")
    for p in principles:
        s = p.statement.lower()
        if any(n in s for n in needles):
            return True
    return False


def _default_sections() -> list[str]:
    return ["Summary", "Reasoning", "Answer", "References"]


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------


def _build_procedure_specs(procedures: list[Procedure]) -> list[RewardSpec]:
    specs: list[RewardSpec] = []
    for proc in procedures:
        keywords: list[str] = []
        for step in sorted(proc.steps, key=lambda s: s.order):
            kw = _step_keywords(step.action, max_keywords=1)
            if kw:
                keywords.append(kw[0])
        if not keywords:
            continue
        specs.append(
            RewardSpec(
                name=_to_identifier("procedure", proc.name or proc.id),
                kind=RewardKind.PROCEDURE,
                description=f"Checks adherence to procedure '{proc.name}' ({proc.id}).",
                weight=1.0,
                procedure_ids=[proc.id],
                config={
                    "procedure_id": proc.id,
                    "ordered": True,
                    "ordered_steps": keywords,
                },
            )
        )
    return specs


def _build_safety_spec(anti_patterns: list[AntiPattern]) -> RewardSpec | None:
    if not anti_patterns:
        return None
    phrase_sets: list[list[str]] = []
    ids: list[str] = []
    for ap in anti_patterns:
        words = [w for w in _WORD_RE.findall(ap.description.lower()) if len(w) > 3]
        if not words:
            continue
        # Take the top 3 distinctive words for an AND-match.
        unique: list[str] = []
        for w in words:
            if w not in unique:
                unique.append(w)
            if len(unique) >= 3:
                break
        phrase_sets.append(unique)
        ids.append(ap.id)
    if not phrase_sets:
        return None
    return RewardSpec(
        name="safety_anti_patterns",
        kind=RewardKind.SAFETY,
        description="Penalises completions that exhibit any catalogued anti-pattern.",
        weight=2.0,
        config={
            "anti_pattern_ids": ids,
            "anti_pattern_terms": phrase_sets,
        },
    )


def _build_principle_specs(
    principles: list[Principle],
    *,
    top_k: int,
    judge_model: str | None,
) -> list[RewardSpec]:
    if not principles or top_k <= 0:
        return []
    ranked = sorted(principles, key=lambda p: (-p.weight, p.id))[:top_k]
    specs: list[RewardSpec] = []
    for p in ranked:
        specs.append(
            RewardSpec(
                name=_to_identifier("principle", p.id),
                kind=RewardKind.PRINCIPLE,
                description=f"LLM-judged adherence to principle {p.id}.",
                weight=float(p.weight),
                principle_ids=[p.id],
                config={
                    "principle_id": p.id,
                    "principle_statement": p.statement,
                    "judge_model": judge_model or "",
                },
            )
        )
    return specs


def _build_lexical_spec(vocabulary: list[VocabularyTerm]) -> RewardSpec | None:
    if not vocabulary:
        return None
    terms: list[str] = []
    for v in vocabulary:
        if v.term:
            terms.append(v.term)
    if not terms:
        return None
    return RewardSpec(
        name="lexical_vocabulary",
        kind=RewardKind.LEXICAL,
        description="Rewards use of domain vocabulary.",
        weight=0.5,
        config={
            "required": terms,
            "forbidden": [],
            "case_sensitive": False,
        },
    )


def _build_citation_spec(min_citations: int = 1) -> RewardSpec:
    return RewardSpec(
        name="citation_knowledge_ids",
        kind=RewardKind.CITATION,
        description="Rewards completions that cite at least one [KNW-####] source.",
        weight=1.0,
        config={
            "require_ids": True,
            "min_citations": min_citations,
            "id_pattern": r"\[KNW-\d+\]",
        },
    )


def _build_format_sections_spec(sections: list[str] | None = None) -> RewardSpec:
    secs = sections or _default_sections()
    return RewardSpec(
        name="format_sections",
        kind=RewardKind.FORMAT,
        description="Rewards completions that include the expected markdown sections.",
        weight=0.75,
        config={"sections": secs},
    )


def _build_length_spec(avg_tokens: int) -> RewardSpec:
    sweet = max(120, min(800, avg_tokens))
    return RewardSpec(
        name="length_budget",
        kind=RewardKind.LENGTH,
        description="Triangular length reward that peaks near the average procedure size.",
        weight=0.25,
        config={
            "min_tokens": max(20, sweet // 4),
            "max_tokens": sweet * 3,
            "sweet_spot": sweet,
        },
    )


# ---------------------------------------------------------------------------
# Top-level designer
# ---------------------------------------------------------------------------


def _enforce_budget(
    specs: list[RewardSpec],
    *,
    budget: int,
) -> list[RewardSpec]:
    """Drop the lowest-priority specs first until ``len(specs) <= budget``.

    Within the same priority, drop the lowest-weight specs first; ties broken
    by descending position (later specs lose first) for reproducibility.
    """
    if budget <= 0 or len(specs) <= budget:
        return specs

    indexed = list(enumerate(specs))
    # Sort so the items we want to KEEP come first.
    indexed.sort(
        key=lambda pair: (
            _PRIORITY[pair[1].kind],
            -pair[1].weight,
            pair[0],
        )
    )
    keep_ordered = sorted(indexed[:budget], key=lambda pair: pair[0])
    return [spec for _, spec in keep_ordered]


async def design_rewards(
    golden: GoldenDocument,
    client: LLMClient | None = None,
    *,
    config: RewardConfig,
) -> RewardBundle:
    """Produce a :class:`RewardBundle` for ``golden`` constrained by ``config``.

    ``client`` is accepted for forward compatibility with a future
    LLM-driven designer; this deterministic implementation does not use it.
    """
    del client  # not yet used
    include = set(config.include_kinds or [])
    judge_model = config.judge_model

    specs: list[RewardSpec] = []

    if "safety" in include:
        s = _build_safety_spec(golden.anti_patterns)
        if s is not None:
            specs.append(s)

    if "procedure" in include:
        specs.extend(_build_procedure_specs(golden.procedures))

    if "principle" in include and golden.principles:
        # Reserve roughly half of the budget for principle rewards but keep at
        # least one if possible.
        cap = max(1, config.reward_budget // 2)
        specs.extend(
            _build_principle_specs(
                golden.principles,
                top_k=min(cap, len(golden.principles)),
                judge_model=judge_model,
            )
        )

    if "citation" in include and (
        _principles_asking_for_citations(golden.principles) or golden.knowledge
    ):
        specs.append(_build_citation_spec())

    if "format" in include and (
        _principles_asking_for_structure(golden.principles) or golden.procedures
    ):
        specs.append(_build_format_sections_spec())

    if "lexical" in include:
        s = _build_lexical_spec(golden.vocabulary)
        if s is not None:
            specs.append(s)

    if "length" in include:
        avg = _avg_procedure_token_length(golden.procedures)
        specs.append(_build_length_spec(avg))

    # De-duplicate by name while preserving order.
    seen: OrderedDict[str, RewardSpec] = OrderedDict()
    for s in specs:
        if s.name not in seen:
            seen[s.name] = s
    specs = list(seen.values())

    capped = _enforce_budget(specs, budget=config.reward_budget)

    if len(capped) < len(specs):
        logger.info(
            "design_rewards: capped %d -> %d specs to fit reward_budget=%d",
            len(specs),
            len(capped),
            config.reward_budget,
        )

    return RewardBundle(
        specs=capped,
        golden_role=golden.identity.role,
        version="1",
    )
