"""Tests for the core pydantic data contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gyroscope.core.models import (
    AntiPattern,
    Chunk,
    Document,
    DocumentKind,
    GoldenDocument,
    Identity,
    KnowledgeItem,
    Principle,
    Procedure,
    ProcedureStep,
    RewardKind,
    RewardSpec,
    Trajectory,
    TrajectoryMessage,
    VocabularyTerm,
)


def _golden(**overrides) -> GoldenDocument:
    base = dict(
        identity=Identity(
            role="RICS Quantity Surveyor",
            description="Cost expert",
            mission="Deliver accurate cost advice",
        ),
        principles=[Principle(id="PRN-0001", statement="Be accurate.")],
        procedures=[
            Procedure(
                id="PRC-0001",
                name="Prepare BoQ",
                purpose="Bill of quantities",
                steps=[ProcedureStep(order=1, action="Measure quantities")],
            )
        ],
        knowledge=[KnowledgeItem(id="KNW-0001", statement="NRM2 is the standard.")],
        vocabulary=[VocabularyTerm(term="BoQ", definition="Bill of Quantities")],
        anti_patterns=[
            AntiPattern(
                id="ANT-0001",
                description="Guessing rates.",
                why_bad="Mispricing.",
                correction="Use rate book.",
            )
        ],
        source_documents=["doc.pdf"],
    )
    base.update(overrides)
    return GoldenDocument(**base)


def test_document_kind_enum():
    d = Document(source="x.pdf", kind=DocumentKind.PDF, text="hello")
    assert d.kind == DocumentKind.PDF
    assert d.token_estimate() >= 1


def test_document_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Document(source="x", kind=DocumentKind.TXT, text="hi", bogus=1)


def test_chunk_minimal():
    c = Chunk(id="src-0001", document_source="x.pdf", text="hi", order=0)
    assert c.order == 0


def test_golden_document_roundtrips_to_markdown():
    g = _golden()
    md = g.to_markdown()
    assert "# Identity" in md
    assert "RICS Quantity Surveyor" in md
    assert "# Principles" in md
    assert "PRN-0001" in md
    assert "# Procedures" in md
    assert "Prepare BoQ" in md
    assert "# Knowledge" in md
    assert "# Vocabulary" in md
    assert "# Anti-patterns" in md
    assert "# Sources" in md
    assert "doc.pdf" in md


def test_trajectory_has_required_messages():
    t = Trajectory(
        id="TRJ-0001",
        scenario_id="SCN-0001",
        system="You are a QS.",
        messages=[
            TrajectoryMessage(role="user", content="What is NRM2?"),
            TrajectoryMessage(role="assistant", content="It is..."),
        ],
        tags={"persona": "junior", "difficulty": "easy"},
    )
    assert t.messages[0].role == "user"
    assert t.tags["difficulty"] == "easy"


def test_reward_spec_kinds():
    r = RewardSpec(
        name="reward_format_sections",
        kind=RewardKind.FORMAT,
        description="must contain sections",
        config={"sections": ["Findings", "Recommendation"]},
        principle_ids=["PRN-0001"],
    )
    assert r.kind == RewardKind.FORMAT
    assert r.weight == 1.0
