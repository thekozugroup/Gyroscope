"""Tests for the curation extractors. The LLM is fully mocked."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from gyroscope.core.config import GyroscopeConfig, LLMConfig
from gyroscope.core.models import Chunk
from gyroscope.curation.extractor import (
    extract_anti_patterns,
    extract_identity,
    extract_knowledge,
    extract_principles,
    extract_procedures,
    extract_vocabulary,
)

# ---------------------------------------------------------------------------
# Stub LLM client. Implements the surface the extractors actually call —
# `_config.llm`, `complete_json_array`, `complete_json`.
# ---------------------------------------------------------------------------


@dataclass
class _StubClient:
    array_returns: list[list[dict[str, Any]]]
    object_return: dict[str, Any] | None = None
    array_calls: list[dict[str, Any]] = field(default_factory=list)
    object_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._config = GyroscopeConfig(api_key="test", llm=LLMConfig())
        self._array_iter = iter(self.array_returns)

    @property
    def config(self) -> GyroscopeConfig:
        return self._config

    def model_for(self, role: str) -> str:
        cfg = self._config.llm
        return {
            "curator": cfg.curator_model,
            "swarm": cfg.swarm_model,
            "critic": cfg.critic_model,
            "judge": cfg.judge_model,
        }[role]

    def temperature_for(self, role: str) -> float:
        cfg = self._config.llm
        return {
            "curator": cfg.temperature_curator,
            "swarm": cfg.temperature_swarm,
            "critic": cfg.temperature_critic,
            "judge": cfg.temperature_critic,
        }[role]

    async def complete_json_array(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.array_calls.append(kwargs)
        try:
            return next(self._array_iter)
        except StopIteration:
            return []

    async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        self.object_calls.append(kwargs)
        if self.object_return is None:
            raise AssertionError("complete_json called but no object_return configured")
        return self.object_return


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c-{i:04d}", document_source="src://x", text=f"text {i}", order=i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_identity_uses_object_payload_and_caches_system():
    client = _StubClient(
        array_returns=[],
        object_return={
            "role": "Quantity Surveyor",
            "description": "Practitioner of measured-quantity costing.",
            "mission": "Deliver accurate cost intelligence on construction projects.",
        },
    )
    identity = await extract_identity(_chunks(5), client)
    assert identity.role == "Quantity Surveyor"
    assert "Deliver" in identity.mission
    assert len(client.object_calls) == 1
    assert client.object_calls[0]["cache_system"] is True


@pytest.mark.asyncio
async def test_extract_identity_falls_back_on_empty_chunks():
    client = _StubClient(array_returns=[])
    identity = await extract_identity([], client)
    assert identity.role == "Unspecified Role"
    # No LLM calls when there's nothing to summarise.
    assert client.object_calls == []


# ---------------------------------------------------------------------------
# Principles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_principles_assigns_ids_and_filters_citations():
    chunks = _chunks(3)
    client = _StubClient(
        array_returns=[
            [
                {
                    "statement": "Verify quantities against drawings.",
                    "source_chunk_ids": ["c-0000"],
                    "rationale": "Accuracy.",
                    "weight": 1.5,
                },
                {
                    "statement": "Hallucinated principle citing nothing real.",
                    "source_chunk_ids": ["unknown"],
                },
                {
                    "statement": "Submit weekly progress reports.",
                    "source_chunk_ids": ["c-0001"],
                },
            ]
        ]
    )
    out = await extract_principles(chunks, client, batch_size=10)
    assert len(out) == 2  # ungrounded one is dropped
    assert [p.id for p in out] == ["PRN-0001", "PRN-0002"]
    assert out[0].weight == 1.5
    assert out[0].rationale == "Accuracy."


@pytest.mark.asyncio
async def test_extract_principles_dedups_identical_statements_across_batches():
    chunks = _chunks(40)
    client = _StubClient(
        array_returns=[
            [
                {"statement": "Always check the drawings.", "source_chunk_ids": ["c-0000"]},
            ],
            [
                {"statement": "  Always check the drawings.  ", "source_chunk_ids": ["c-0024"]},
                {"statement": "Use the BoQ to drive procurement.", "source_chunk_ids": ["c-0025"]},
            ],
        ]
    )
    out = await extract_principles(chunks, client, batch_size=24)
    assert len(out) == 2
    assert {p.statement for p in out} == {
        "Always check the drawings.",
        "Use the BoQ to drive procurement.",
    }
    # Two batches were issued.
    assert len(client.array_calls) == 2
    # System prompt was identical across batches => cache_system=True.
    assert client.array_calls[0]["system"] == client.array_calls[1]["system"]
    assert all(call["cache_system"] is True for call in client.array_calls)


# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_procedures_builds_steps():
    chunks = _chunks(2)
    client = _StubClient(
        array_returns=[
            [
                {
                    "name": "Issue an RFI",
                    "purpose": "Resolve ambiguity in the design.",
                    "steps": [
                        {"order": 1, "action": "Identify the ambiguity", "expected_output": None},
                        {"order": 2, "action": "Draft RFI", "tool": "rfi_tool"},
                        {"order": 3, "action": "Send to designer"},
                    ],
                    "source_chunk_ids": ["c-0000"],
                },
                {
                    # Should be dropped: no steps.
                    "name": "Empty",
                    "purpose": "Nothing",
                    "steps": [],
                    "source_chunk_ids": ["c-0001"],
                },
            ]
        ]
    )
    out, dropped = await extract_procedures(chunks, client, batch_size=10)
    assert len(out) == 1
    assert dropped == 1  # the empty-steps procedure was dropped
    proc = out[0]
    assert proc.id == "PRC-0001"
    assert [s.order for s in proc.steps] == [1, 2, 3]
    assert proc.steps[1].tool == "rfi_tool"


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_knowledge_assigns_ids_and_keeps_tags():
    chunks = _chunks(2)
    client = _StubClient(
        array_returns=[
            [
                {
                    "statement": "Concrete typically reaches design strength after 28 days.",
                    "citations": ["c-0000"],
                    "tags": ["concrete", "curing"],
                },
                {
                    "statement": "Floating fact citing unknown chunk.",
                    "citations": ["c-9999"],
                },
            ]
        ]
    )
    out = await extract_knowledge(chunks, client, batch_size=10)
    assert len(out) == 1
    assert out[0].id == "KNW-0001"
    assert out[0].tags == ["concrete", "curing"]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_vocabulary_dedups_case_insensitively():
    chunks = _chunks(2)
    client = _StubClient(
        array_returns=[
            [
                {"term": "BoQ", "definition": "Bill of Quantities"},
                {"term": "boq", "definition": "duplicate"},
                {"term": "RFI", "definition": "Request for Information", "aliases": ["query"]},
                {"term": "", "definition": "skip me"},
            ]
        ]
    )
    out = await extract_vocabulary(chunks, client, batch_size=10)
    assert {t.term for t in out} == {"BoQ", "RFI"}
    rfi = next(t for t in out if t.term == "RFI")
    assert rfi.aliases == ["query"]


# ---------------------------------------------------------------------------
# Anti-patterns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_anti_patterns_assigns_ids():
    chunks = _chunks(2)
    client = _StubClient(
        array_returns=[
            [
                {
                    "description": "Skipping site verification before pour",
                    "why_bad": "Leads to defects",
                    "correction": "Always conduct a pre-pour check",
                    "source_chunk_ids": ["c-0000"],
                },
            ]
        ]
    )
    out = await extract_anti_patterns(chunks, client, batch_size=10)
    assert len(out) == 1
    assert out[0].id == "ANT-0001"
    assert out[0].source_chunk_ids == ["c-0000"]


# ---------------------------------------------------------------------------
# Batching: ensures we issue multiple calls when chunks > batch_size.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extractor_batches_chunks_per_request():
    chunks = _chunks(7)
    client = _StubClient(array_returns=[[], [], []])  # 3 batches expected
    await extract_principles(chunks, client, batch_size=3)
    assert len(client.array_calls) == 3
