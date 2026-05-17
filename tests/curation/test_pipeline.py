"""End-to-end test for CurationPipeline with the LLM fully mocked."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from gyroscope.core.config import CurationConfig, GyroscopeConfig, LLMConfig
from gyroscope.core.models import Document, DocumentKind, GoldenDocument
from gyroscope.curation.pipeline import CurationPipeline


@dataclass
class _StubClient:
    """Replays canned payloads. ``complete_json`` returns one Identity
    object; ``complete_json_array`` cycles per-extractor through a
    deterministic queue."""

    identity_payload: dict[str, Any]
    array_queue: list[list[dict[str, Any]]]
    array_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._config = GyroscopeConfig(api_key="test", llm=LLMConfig())
        self._iter = iter(self.array_queue)

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

    async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        return dict(self.identity_payload)

    async def complete_json_array(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.array_calls.append(kwargs)
        try:
            return next(self._iter)
        except StopIteration:
            return []


def _make_document(source: str, text: str) -> Document:
    return Document(source=source, kind=DocumentKind.MARKDOWN, text=text)


@pytest.mark.asyncio
async def test_curation_pipeline_end_to_end(tmp_path: Path):
    docs = [
        _make_document(
            "memo://doc-a",
            "## Intro\n\nThe role oversees quality on construction sites.\n\n"
            "## Duties\n\nThe surveyor measures works and certifies payments.",
        ),
        _make_document(
            "memo://doc-b",
            "## Procedures\n\nFollow the standard inspection routine on every visit.\n\n"
            "## Notes\n\nKeep records of variations.",
        ),
    ]

    # Identity payload — single object.
    identity = {
        "role": "Site Quality Surveyor",
        "description": "Oversees workmanship and measurement on construction sites.",
        "mission": "Ensure works conform to specification and quantities are accurate.",
    }

    # We do not know how many batches the chunker will produce, but the
    # stub gracefully returns [] when the queue is exhausted. We seed one
    # payload per extractor and let trailing batches return empty.
    array_queue: list[list[dict[str, Any]]] = [
        # principles batch 1
        [
            {
                "statement": "Measure works accurately before certifying payment.",
                "source_chunk_ids": ["__first_chunk__"],
                "rationale": "Prevents over-payment.",
            },
            {
                "statement": "Keep records of every variation order.",
                "source_chunk_ids": ["__first_chunk__"],
            },
        ],
        # procedures batch 1
        [
            {
                "name": "Standard inspection",
                "purpose": "Verify works on site.",
                "steps": [
                    {"order": 1, "action": "Walk the site"},
                    {"order": 2, "action": "Record defects"},
                ],
                "source_chunk_ids": ["__first_chunk__"],
            }
        ],
        # knowledge batch 1
        [
            {
                "statement": "Variations must be agreed in writing before execution.",
                "citations": ["__first_chunk__"],
                "tags": ["variations"],
            }
        ],
        # vocabulary batch 1
        [
            {
                "term": "Variation",
                "definition": "An authorised change to the scope of the contracted works.",
            }
        ],
        # anti-patterns batch 1
        [
            {
                "description": "Certifying payment without site measurement.",
                "why_bad": "Risks over-payment and disputes.",
                "correction": "Always measure on site first.",
                "source_chunk_ids": ["__first_chunk__"],
            }
        ],
    ]

    client = _StubClient(identity_payload=identity, array_queue=array_queue)

    cfg = GyroscopeConfig(
        api_key="test",
        output_dir=tmp_path / "out",
        curation=CurationConfig(chunk_target_tokens=20),
        llm=LLMConfig(),
    )
    pipeline = CurationPipeline(cfg)

    # Patch the placeholder source chunk id with the first real chunk id
    # the pipeline will produce. We do this by replaying a tiny prefix of
    # the chunker so the stub's ids align with reality.
    from gyroscope.curation.chunker import SemanticChunker

    chunks = SemanticChunker(cfg.curation).chunk_documents(docs)
    assert chunks, "expected the chunker to produce at least one chunk"
    real_first_id = chunks[0].id
    for batch in array_queue:
        for item in batch:
            for key in ("source_chunk_ids", "citations"):
                if key in item:
                    item[key] = [
                        real_first_id if cid == "__first_chunk__" else cid for cid in item[key]
                    ]

    golden = await pipeline.distill(docs, client)

    # GoldenDocument is valid.
    assert isinstance(golden, GoldenDocument)
    assert golden.identity.role == "Site Quality Surveyor"
    assert len(golden.principles) == 2
    assert [p.id for p in golden.principles] == ["PRN-0001", "PRN-0002"]
    assert len(golden.procedures) == 1
    assert golden.procedures[0].id == "PRC-0001"
    assert len(golden.knowledge) == 1
    assert golden.knowledge[0].id == "KNW-0001"
    assert len(golden.vocabulary) == 1
    assert len(golden.anti_patterns) == 1
    assert golden.anti_patterns[0].id == "ANT-0001"
    assert golden.source_documents == ["memo://doc-a", "memo://doc-b"]

    # Markdown renders.
    md = golden.to_markdown()
    assert "# Identity" in md
    assert "# Principles" in md
    assert "# Procedures" in md
    assert "# Knowledge" in md
    assert "# Vocabulary" in md
    assert "# Anti-patterns" in md
    assert "[PRN-0001]" in md

    # Files were written.
    md_path = cfg.output_dir / "golden.md"
    json_path = cfg.output_dir / "golden.json"
    assert md_path.exists()
    assert json_path.exists()
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk["identity"]["role"] == "Site Quality Surveyor"
    assert md_path.read_text(encoding="utf-8") == md


@pytest.mark.asyncio
async def test_pipeline_rejects_empty_corpus(tmp_path: Path):
    cfg = GyroscopeConfig(api_key="test", output_dir=tmp_path)
    pipeline = CurationPipeline(cfg)
    client = _StubClient(identity_payload={"role": "x", "description": "y", "mission": "z"}, array_queue=[])
    with pytest.raises(ValueError):
        await pipeline.distill([], client)
