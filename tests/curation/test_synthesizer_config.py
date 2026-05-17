"""Synthesizer respects the configured dedup threshold.

Spec: ``CurationConfig.synthesizer_dedup_threshold`` is the knob users tune
to control near-duplicate collapse of principle / knowledge statements after
extraction. Same input + different threshold must change the result.
"""

from __future__ import annotations

import pytest

from gyroscope.core.config import CurationConfig
from gyroscope.core.models import Identity, KnowledgeItem, Principle
from gyroscope.curation.synthesizer import ExtractorOutputs, synthesize


def _identity() -> Identity:
    return Identity(role="Test Role", description="d", mission="m")


def _make_extracts(
    *,
    principles: list[Principle] | None = None,
    knowledge: list[KnowledgeItem] | None = None,
) -> ExtractorOutputs:
    return ExtractorOutputs(
        identity=_identity(),
        principles=list(principles or []),
        procedures=[],
        knowledge=list(knowledge or []),
        vocabulary=[],
        anti_patterns=[],
        source_documents=["src://x"],
    )


def _principle_pair() -> list[Principle]:
    """Two near-duplicates whose tokenised Jaccard sits between 0.8 and 0.9.

    Computing by hand on lower-cased tokens:
      A = {always, verify, quantities, against, drawings}
      B = {always, verify, quantities, against, the, drawings}
      |A intersect B| = 5, |A union B| = 6 -> Jaccard = 5/6 ~= 0.833.

    So at threshold 0.8 these two collapse; at threshold 0.9 they survive
    as distinct principles.
    """
    return [
        Principle(
            id="PRN-A",
            statement="Always verify quantities against drawings.",
            source_chunk_ids=["c1"],
        ),
        Principle(
            id="PRN-B",
            statement="Always verify quantities against the drawings.",
            source_chunk_ids=["c2"],
        ),
    ]


def _knowledge_pair() -> list[KnowledgeItem]:
    """Same construction as ``_principle_pair`` but for KnowledgeItem.

    A = {concrete, sets, in, 28, days}                 -> 5 tokens
    B = {concrete, sets, hard, in, 28, days}           -> 6 tokens
    |A intersect B| = 5, |A union B| = 6 -> 5/6 ~= 0.833.
    """
    return [
        KnowledgeItem(
            id="KNW-A",
            statement="Concrete sets in 28 days.",
            citations=["c1"],
        ),
        KnowledgeItem(
            id="KNW-B",
            statement="Concrete sets hard in 28 days.",
            citations=["c2"],
        ),
    ]


@pytest.mark.asyncio
async def test_synthesizer_collapses_near_duplicate_principles_at_low_threshold():
    extracts = _make_extracts(principles=_principle_pair())
    cfg = CurationConfig(synthesizer_dedup_threshold=0.8)

    golden = await synthesize(extracts, client=None, config=cfg)

    assert len(golden.principles) == 1
    # Citations from the dropped duplicate were merged in.
    survivor = golden.principles[0]
    assert set(survivor.source_chunk_ids) == {"c1", "c2"}


@pytest.mark.asyncio
async def test_synthesizer_keeps_near_duplicate_principles_at_high_threshold():
    extracts = _make_extracts(principles=_principle_pair())
    cfg = CurationConfig(synthesizer_dedup_threshold=0.9)

    golden = await synthesize(extracts, client=None, config=cfg)

    # Two near-duplicates survive when the threshold is stricter than their
    # Jaccard similarity.
    assert len(golden.principles) == 2
    # Each survivor keeps only its own original citation.
    citation_sets = [frozenset(p.source_chunk_ids) for p in golden.principles]
    assert set(citation_sets) == {frozenset({"c1"}), frozenset({"c2"})}


@pytest.mark.asyncio
async def test_synthesizer_threshold_applies_to_knowledge_too():
    extracts = _make_extracts(knowledge=_knowledge_pair())

    aggressive = await synthesize(
        extracts, client=None, config=CurationConfig(synthesizer_dedup_threshold=0.8)
    )
    strict = await synthesize(
        extracts, client=None, config=CurationConfig(synthesizer_dedup_threshold=0.9)
    )

    assert len(aggressive.knowledge) == 1
    assert set(aggressive.knowledge[0].citations) == {"c1", "c2"}
    assert len(strict.knowledge) == 2


@pytest.mark.asyncio
async def test_default_config_preserves_legacy_synthesizer_behaviour():
    """The default ``CurationConfig()`` must keep the historical 0.8 cut-off
    so existing pipelines see no behaviour change after the wiring."""
    extracts = _make_extracts(principles=_principle_pair())
    golden = await synthesize(extracts, client=None, config=CurationConfig())
    assert len(golden.principles) == 1
