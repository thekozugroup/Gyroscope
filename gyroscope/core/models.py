"""Pydantic data contracts shared across all Gyroscope phases.

These are the only types that cross phase boundaries. Every downstream stage
consumes and produces instances of these models so that a phase can be
swapped, mocked, or re-run independently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class DocumentKind(StrEnum):
    PDF = "pdf"
    WEB = "web"
    MARKDOWN = "md"
    HTML = "html"
    DOCX = "docx"
    TXT = "txt"


class Document(BaseModel):
    """A single source document after ingestion."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="Origin path or URL.")
    kind: DocumentKind
    text: str = Field(..., description="Cleaned plain text or markdown.")
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def token_estimate(self) -> int:
        """Cheap len/4 estimate, useful before paying for a real tokenizer."""
        return max(1, len(self.text) // 4)


class Chunk(BaseModel):
    """A semantic chunk of a Document used during curation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    document_source: str
    text: str
    order: int
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Golden Document — output of curation, input of SFT and reward design
# ---------------------------------------------------------------------------


class Identity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(..., description="One-line role title, e.g. 'RICS Quantity Surveyor'.")
    description: str
    mission: str


class Principle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable id, e.g. PRN-0001.")
    statement: str = Field(..., description="Atomic, testable principle.")
    rationale: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    weight: float = 1.0


class ProcedureStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int
    action: str
    expected_output: str | None = None
    tool: str | None = None


class Procedure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable id, e.g. PRC-0001.")
    name: str
    purpose: str
    steps: list[ProcedureStep]
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class KnowledgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable id, e.g. KNW-0001.")
    statement: str
    citations: list[str] = Field(default_factory=list, description="Source chunk ids.")
    tags: list[str] = Field(default_factory=list)


class VocabularyTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    definition: str
    aliases: list[str] = Field(default_factory=list)


class AntiPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable id, e.g. ANT-0001.")
    description: str
    why_bad: str
    correction: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class GoldenDocument(BaseModel):
    """The single distilled artefact used by SFT and reward design."""

    model_config = ConfigDict(extra="forbid")

    identity: Identity
    principles: list[Principle] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    knowledge: list[KnowledgeItem] = Field(default_factory=list)
    vocabulary: list[VocabularyTerm] = Field(default_factory=list)
    anti_patterns: list[AntiPattern] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_markdown(self) -> str:
        """Render the golden document as the canonical markdown format."""
        from gyroscope.core.io import golden_to_markdown

        return golden_to_markdown(self)


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    expertise_level: Literal["novice", "intermediate", "expert"] = "intermediate"
    tone: str = "neutral"


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    procedure_id: str | None = None
    principle_ids: list[str] = Field(default_factory=list)
    persona_id: str
    difficulty: Literal["easy", "medium", "hard", "adversarial"] = "medium"
    prompt_seed: str


class TrajectoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None  # tool name when role == "tool"


class Trajectory(BaseModel):
    """One SFT example. Multi-turn, principle-tagged."""

    model_config = ConfigDict(extra="forbid")

    id: str
    scenario_id: str
    system: str
    messages: list[TrajectoryMessage]
    tags: dict[str, Any] = Field(default_factory=dict)
    quality_score: float | None = None
    critic_notes: str | None = None

    def to_sharegpt(self) -> dict[str, Any]:
        """Serialise to ShareGPT row."""
        from gyroscope.sft.formats import to_sharegpt

        return to_sharegpt(self)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


class RewardKind(StrEnum):
    FORMAT = "format"
    LEXICAL = "lexical"
    PRINCIPLE = "principle"
    PROCEDURE = "procedure"
    SAFETY = "safety"
    CITATION = "citation"
    LENGTH = "length"


class RewardSpec(BaseModel):
    """Declarative spec for a single reward function."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Python-safe identifier.")
    kind: RewardKind
    description: str
    weight: float = 1.0
    principle_ids: list[str] = Field(default_factory=list)
    procedure_ids: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    """kind-specific config:
       format:    {pattern: regex} | {json_schema: dict} | {sections: [str]}
       lexical:   {required: [str], forbidden: [str], case_sensitive: bool}
       principle: {principle_id: str, judge_model: str}
       procedure: {procedure_id: str, ordered: bool}
       safety:    {anti_pattern_ids: [str]}
       citation:  {require_ids: bool, min_citations: int}
       length:    {min_tokens: int, max_tokens: int, sweet_spot: int}
    """
