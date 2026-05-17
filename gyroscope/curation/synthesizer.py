"""Merge the parallel extractor outputs into a single GoldenDocument.

The synthesizer:

* Performs embedding-free near-duplicate detection on Principle and
  KnowledgeItem statements using token-set Jaccard.
* Enforces the per-category caps from :class:`CurationConfig`,
  prioritising items with the most ``source_chunk_ids`` / ``citations``
  (more grounded = more important).
* Re-assigns deterministic ids so the final document is densely numbered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gyroscope.core.config import CurationConfig
from gyroscope.core.llm import LLMClient
from gyroscope.core.models import (
    AntiPattern,
    GoldenDocument,
    Identity,
    KnowledgeItem,
    Principle,
    Procedure,
    VocabularyTerm,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Items with Jaccard at or above this on their normalised statements are
# treated as semantic duplicates and collapsed.
_DEDUP_JACCARD = 0.8


@dataclass(frozen=True)
class ExtractorOutputs:
    """Bundle of all extractor results passed into the synthesizer."""

    identity: Identity
    principles: list[Principle]
    procedures: list[Procedure]
    knowledge: list[KnowledgeItem]
    vocabulary: list[VocabularyTerm]
    anti_patterns: list[AntiPattern]
    source_documents: list[str]


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _dedup_by_statement(
    items: list[Principle] | list[KnowledgeItem],
    *,
    statement_attr: str,
    citations_attr: str,
) -> list:
    """Greedy near-duplicate collapse: walk items in order, keep an item
    only when its tokenised statement is below the Jaccard threshold
    against every already-kept item. When collapsing, merge the dropped
    item's citation ids into the survivor."""
    kept: list = []
    kept_tokens: list[set[str]] = []
    for item in items:
        statement = getattr(item, statement_attr)
        toks = _tokens(statement)
        merged_into: int | None = None
        for idx, prior_toks in enumerate(kept_tokens):
            if _jaccard(toks, prior_toks) >= _DEDUP_JACCARD:
                merged_into = idx
                break
        if merged_into is None:
            kept.append(item)
            kept_tokens.append(toks)
            continue
        # Merge citations and union with survivor.
        survivor = kept[merged_into]
        existing = list(getattr(survivor, citations_attr))
        for cid in getattr(item, citations_attr):
            if cid not in existing:
                existing.append(cid)
        setattr(survivor, citations_attr, existing)
    return kept


def _dedup_vocabulary(items: list[VocabularyTerm]) -> list[VocabularyTerm]:
    """Merge duplicate vocabulary terms by case-insensitive term match."""
    by_key: dict[str, VocabularyTerm] = {}
    for term in items:
        key = term.term.strip().lower()
        if not key:
            continue
        if key not in by_key:
            by_key[key] = term
            continue
        # Merge aliases.
        survivor = by_key[key]
        aliases = list(survivor.aliases)
        for a in term.aliases:
            if a not in aliases:
                aliases.append(a)
        survivor.aliases = aliases
    return list(by_key.values())


def _prioritise(
    items: list,
    *,
    citation_key: str,
    cap: int,
) -> list:
    """Return up to ``cap`` items, sorted by len(citation_key) descending,
    breaking ties by original order so the result remains deterministic."""
    if cap <= 0 or len(items) <= cap:
        return list(items)
    decorated = [
        (-len(getattr(item, citation_key) or []), i, item) for i, item in enumerate(items)
    ]
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in decorated[:cap]]


def _renumber(items: list, *, prefix: str) -> list:
    """Renumber items deterministically. Mutates each pydantic model."""
    for i, item in enumerate(items, start=1):
        item.id = f"{prefix}-{i:04d}"
    return items


async def synthesize(
    extracts: ExtractorOutputs,
    client: LLMClient,
    *,
    config: CurationConfig | None = None,
) -> GoldenDocument:
    """Merge extractor outputs into a coherent GoldenDocument.

    ``client`` is accepted for API symmetry / future use (it is part of the
    documented signature) but the current implementation merges without
    additional LLM calls — everything we need is already in ``extracts``.
    """
    cfg = config or CurationConfig()

    # Principle dedup + cap + renumber.
    principles = _dedup_by_statement(
        list(extracts.principles),
        statement_attr="statement",
        citations_attr="source_chunk_ids",
    )
    principles = _prioritise(
        principles, citation_key="source_chunk_ids", cap=cfg.max_principles
    )
    principles = _renumber(list(principles), prefix="PRN")

    # Procedures: dedup by (name, len(steps)) preserving order, then cap by
    # most-cited.
    seen_proc: set[tuple[str, int]] = set()
    deduped_procs: list[Procedure] = []
    for proc in extracts.procedures:
        key = (proc.name.strip().lower(), len(proc.steps))
        if key in seen_proc:
            continue
        seen_proc.add(key)
        deduped_procs.append(proc)
    procedures = _prioritise(
        deduped_procs, citation_key="source_chunk_ids", cap=cfg.max_procedures
    )
    procedures = _renumber(list(procedures), prefix="PRC")

    # Knowledge dedup + cap + renumber.
    knowledge = _dedup_by_statement(
        list(extracts.knowledge),
        statement_attr="statement",
        citations_attr="citations",
    )
    knowledge = _prioritise(
        knowledge, citation_key="citations", cap=cfg.max_knowledge_items
    )
    knowledge = _renumber(list(knowledge), prefix="KNW")

    # Vocabulary dedup. No cap requested by spec.
    vocabulary = _dedup_vocabulary(list(extracts.vocabulary))

    # Anti-patterns: dedup by description, renumber.
    seen_anti: set[str] = set()
    deduped_anti: list[AntiPattern] = []
    for ap in extracts.anti_patterns:
        key = ap.description.strip().lower()
        if not key or key in seen_anti:
            continue
        seen_anti.add(key)
        deduped_anti.append(ap)
    anti_patterns = _renumber(list(deduped_anti), prefix="ANT")

    return GoldenDocument(
        identity=extracts.identity,
        principles=principles,
        procedures=procedures,
        knowledge=knowledge,
        vocabulary=vocabulary,
        anti_patterns=anti_patterns,
        source_documents=list(extracts.source_documents),
    )
