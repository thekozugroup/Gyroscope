"""Tests for the curation synthesizer."""

from __future__ import annotations

import pytest

from gyroscope.core.config import CurationConfig
from gyroscope.core.models import (
    AntiPattern,
    Identity,
    KnowledgeItem,
    Principle,
    Procedure,
    ProcedureStep,
    VocabularyTerm,
)
from gyroscope.curation.synthesizer import ExtractorOutputs, synthesize


def _identity() -> Identity:
    return Identity(role="Test Role", description="d", mission="m")


def _make_extracts(**overrides) -> ExtractorOutputs:
    base = {
        "identity": _identity(),
        "principles": [],
        "procedures": [],
        "knowledge": [],
        "vocabulary": [],
        "anti_patterns": [],
        "source_documents": ["src://x"],
    }
    base.update(overrides)
    return ExtractorOutputs(**base)


@pytest.mark.asyncio
async def test_synthesize_enforces_principle_cap_and_prefers_grounded():
    # 5 principles, cap=3. Survivors should be the three most-cited.
    principles = [
        Principle(id="PRN-0001", statement="alpha rule one", source_chunk_ids=["c1"]),
        Principle(
            id="PRN-0002",
            statement="bravo rule two",
            source_chunk_ids=["c1", "c2", "c3"],
        ),
        Principle(id="PRN-0003", statement="charlie rule three", source_chunk_ids=["c1", "c2"]),
        Principle(id="PRN-0004", statement="delta rule four", source_chunk_ids=[]),
        Principle(id="PRN-0005", statement="echo rule five", source_chunk_ids=["c5"]),
    ]
    extracts = _make_extracts(principles=principles)
    cfg = CurationConfig(max_principles=3)

    golden = await synthesize(extracts, client=None, config=cfg)

    assert len(golden.principles) == 3
    statements = [p.statement for p in golden.principles]
    assert "bravo rule two" in statements
    assert "charlie rule three" in statements
    # The two-citation 'charlie' must beat the single-citation entries.
    assert "delta rule four" not in statements
    # Re-numbered densely.
    assert [p.id for p in golden.principles] == ["PRN-0001", "PRN-0002", "PRN-0003"]


@pytest.mark.asyncio
async def test_synthesize_dedups_near_duplicate_principles():
    principles = [
        Principle(
            id="PRN-A",
            statement="Always verify quantities against drawings.",
            source_chunk_ids=["c1"],
        ),
        Principle(
            id="PRN-B",
            statement="Always verify quantities against the drawings",
            source_chunk_ids=["c2"],
        ),
        Principle(
            id="PRN-C",
            statement="Submit weekly progress reports to the project manager.",
            source_chunk_ids=["c3"],
        ),
    ]
    extracts = _make_extracts(principles=principles)
    golden = await synthesize(extracts, client=None, config=CurationConfig())

    assert len(golden.principles) == 2
    survivor = golden.principles[0]
    # Citations from the merged duplicate were carried into the survivor.
    assert "c1" in survivor.source_chunk_ids
    assert "c2" in survivor.source_chunk_ids


@pytest.mark.asyncio
async def test_synthesize_caps_procedures_and_knowledge():
    procedures = [
        Procedure(
            id=f"PRC-{i:04d}",
            name=f"Procedure {i}",
            purpose="p",
            steps=[ProcedureStep(order=1, action="do thing")],
            source_chunk_ids=["c"] * (i + 1),
        )
        for i in range(5)
    ]
    knowledge = [
        KnowledgeItem(
            id=f"KNW-{i:04d}",
            statement=f"Fact number {i}",
            citations=["c"] * ((i % 3) + 1),
        )
        for i in range(7)
    ]
    extracts = _make_extracts(procedures=procedures, knowledge=knowledge)
    cfg = CurationConfig(max_procedures=2, max_knowledge_items=3)
    golden = await synthesize(extracts, client=None, config=cfg)

    assert len(golden.procedures) == 2
    assert [p.id for p in golden.procedures] == ["PRC-0001", "PRC-0002"]
    assert len(golden.knowledge) == 3
    assert [k.id for k in golden.knowledge] == ["KNW-0001", "KNW-0002", "KNW-0003"]


@pytest.mark.asyncio
async def test_synthesize_merges_vocabulary_aliases():
    vocab = [
        VocabularyTerm(term="QS", definition="Quantity Surveyor", aliases=["surveyor"]),
        VocabularyTerm(term="qs", definition="Quantity Surveyor (duplicate)", aliases=["measurer"]),
    ]
    # Vocabulary-only corpora trip the empty-BoK invariant (no principle /
    # procedure / knowledge has anything to ground), so pair the vocab with
    # one principle just to keep this test focused on the vocab merge logic.
    extracts = _make_extracts(
        principles=[
            Principle(id="PRN-X", statement="One rule.", source_chunk_ids=["c1"]),
        ],
        vocabulary=vocab,
    )
    golden = await synthesize(extracts, client=None, config=CurationConfig())
    assert len(golden.vocabulary) == 1
    survivor = golden.vocabulary[0]
    assert "surveyor" in survivor.aliases
    assert "measurer" in survivor.aliases


@pytest.mark.asyncio
async def test_synthesize_round_trips_through_markdown():
    extracts = _make_extracts(
        principles=[
            Principle(id="PRN-X", statement="One rule.", source_chunk_ids=["c1"]),
        ],
        anti_patterns=[
            AntiPattern(
                id="ANT-X",
                description="Skipping reviews",
                why_bad="Errors slip through",
                correction="Always review",
                source_chunk_ids=["c1"],
            )
        ],
    )
    golden = await synthesize(extracts, client=None, config=CurationConfig())
    md = golden.to_markdown()
    assert "# Identity" in md
    assert "[PRN-0001]" in md
    assert "[ANT-0001]" in md
