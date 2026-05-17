"""Empty-input invariants for the curation phase.

The synthesizer must refuse to emit a useless GoldenDocument: if Identity
has no role, *or* every extractable category is empty, every downstream
phase is broken — so we fail loud (``EmptyBoKError``) rather than silently
write an empty ``golden.md``.

The extractor must also surface the per-call count of zero-step procedures
it dropped so the curation pipeline can include that telemetry in its run
report. Earlier revisions exposed this via a module-level global; we now
return it as the second element of the ``extract_procedures`` tuple to
avoid races when two extractions run concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from gyroscope.core.config import CurationConfig, GyroscopeConfig, LLMConfig
from gyroscope.core.models import Chunk, Identity, Principle
from gyroscope.curation.extractor import extract_procedures
from gyroscope.curation.synthesizer import (
    EmptyBoKError,
    ExtractorOutputs,
    synthesize,
)

# ---------------------------------------------------------------------------
# EmptyBoKError
# ---------------------------------------------------------------------------


def _empty_extracts(*, role: str = "Stub Role") -> ExtractorOutputs:
    return ExtractorOutputs(
        identity=Identity(role=role, description="d", mission="m"),
        principles=[],
        procedures=[],
        knowledge=[],
        vocabulary=[],
        anti_patterns=[],
        source_documents=["src://empty"],
    )


@pytest.mark.asyncio
async def test_synthesize_raises_on_no_extractable_content():
    """All three of principles/procedures/knowledge empty => EmptyBoKError."""
    extracts = _empty_extracts()
    with pytest.raises(EmptyBoKError) as excinfo:
        await synthesize(extracts, client=None, config=CurationConfig())
    # Message should be operator-actionable — name the empty corpus.
    msg = str(excinfo.value)
    assert "principles=0" in msg
    assert "procedures=0" in msg
    assert "knowledge=0" in msg


@pytest.mark.asyncio
async def test_synthesize_raises_on_empty_role_even_with_content():
    """An Identity with an empty role is unusable downstream."""
    extracts = ExtractorOutputs(
        identity=Identity(role="", description="d", mission="m"),
        principles=[Principle(id="PRN-X", statement="A real principle.", source_chunk_ids=["c1"])],
        procedures=[],
        knowledge=[],
        vocabulary=[],
        anti_patterns=[],
        source_documents=["src://x"],
    )
    with pytest.raises(EmptyBoKError):
        await synthesize(extracts, client=None, config=CurationConfig())


def test_empty_bok_error_is_runtime_error_subclass():
    """Callers catching RuntimeError must also catch EmptyBoKError."""
    assert issubclass(EmptyBoKError, RuntimeError)


# ---------------------------------------------------------------------------
# extract_procedures dropped-count telemetry
# ---------------------------------------------------------------------------


@dataclass
class _StubClient:
    array_returns: list[list[dict[str, Any]]]
    array_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._config = GyroscopeConfig(api_key="test", llm=LLMConfig())
        self._array_iter = iter(self.array_returns)

    # The extractor now reads model/temperature via these helpers (added by
    # the LLMClient refactor). Provide both legacy ``_config`` access and
    # the new ``model_for`` / ``temperature_for`` surface so the stub keeps
    # working across both code paths.
    def model_for(self, role: str) -> str:
        return self._config.llm.curator_model

    def temperature_for(self, role: str) -> float:
        return self._config.llm.temperature_curator

    async def complete_json_array(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.array_calls.append(kwargs)
        try:
            return next(self._array_iter)
        except StopIteration:
            return []


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(id=f"c-{i:04d}", document_source="src://x", text=f"text {i}", order=i)
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_extract_procedures_counts_dropped_zero_step_procedures():
    chunks = _chunks(2)
    client = _StubClient(
        array_returns=[
            [
                {
                    "name": "Good procedure",
                    "purpose": "p",
                    "steps": [{"order": 1, "action": "do thing"}],
                    "source_chunk_ids": ["c-0000"],
                },
                {
                    "name": "Empty A",
                    "purpose": "p",
                    "steps": [],
                    "source_chunk_ids": ["c-0001"],
                },
                {
                    "name": "Empty B",
                    "purpose": "p",
                    "steps": [],
                    "source_chunk_ids": ["c-0000"],
                },
            ]
        ]
    )
    out, dropped = await extract_procedures(chunks, client, batch_size=10)
    assert len(out) == 1
    # Two procedures had zero steps and were dropped this call.
    assert dropped == 2


@pytest.mark.asyncio
async def test_extract_procedures_dropped_count_is_per_call():
    """Each call reports only the procedures it dropped, not a cumulative total."""
    # First call drops one.
    chunks = _chunks(1)
    client1 = _StubClient(
        array_returns=[
            [
                {
                    "name": "Empty",
                    "purpose": "p",
                    "steps": [],
                    "source_chunk_ids": ["c-0000"],
                },
            ]
        ]
    )
    _, dropped1 = await extract_procedures(chunks, client1, batch_size=10)
    assert dropped1 == 1

    # Second call drops zero.
    client2 = _StubClient(
        array_returns=[
            [
                {
                    "name": "Real",
                    "purpose": "p",
                    "steps": [{"order": 1, "action": "do"}],
                    "source_chunk_ids": ["c-0000"],
                },
            ]
        ]
    )
    _, dropped2 = await extract_procedures(chunks, client2, batch_size=10)
    assert dropped2 == 0
