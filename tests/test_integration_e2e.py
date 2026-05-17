"""End-to-end integration test.

Wires the REAL implementations of every phase together (ingestion → curation
→ sft → rewards → quality report) with the LLM client fully mocked. Catches
cross-phase contract regressions that per-phase unit tests miss.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from gyroscope.core.config import GyroscopeConfig
from gyroscope.core.models import GoldenDocument
from gyroscope.quality.metrics import assemble_report

_CHUNK_ID_RE = re.compile(r'<<CHUNK id="([^"]+)"')


class FakeLLM:
    """A non-network LLMClient compatible enough for every phase to run.

    All async entry points are implemented and return deterministic payloads
    that satisfy the JSON shapes each phase parses. The point is to exercise
    composition and contracts, not to evaluate LLM behaviour.
    """

    def __init__(self, config: GyroscopeConfig) -> None:
        self._config = config

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

    async def __aenter__(self) -> FakeLLM:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aclose(self) -> None:
        return None

    # generic complete is rarely used by curation/sft but exists for parity
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool | None = None,
        assistant_prefill: str | None = None,
    ) -> str:
        if assistant_prefill == "{":
            return assistant_prefill + json.dumps(self._infer_object(user))[1:]
        if assistant_prefill == "[":
            return assistant_prefill + json.dumps(self._infer_array(user))[1:]
        return "OK."

    async def complete_messages(self, *, system: str, messages, **kwargs: Any) -> str:
        last_user = ""
        for m in reversed(messages):
            if getattr(m, "role", None) == "user":
                last_user = getattr(m, "content", "")
                break
        if "END" in system.upper() or "end conversation" in (last_user or "").lower():
            return "<END>"
        if "user" in system.lower():
            return "Could you walk me through the next step?"
        return (
            "Per the procedure, the next action is to measure quantities and cite "
            "[KNW-0001]. Findings: deterministic. Recommendation: proceed."
        )

    async def complete_json(self, *, system: str, user: str, **kwargs: Any) -> Any:
        return self._infer_object(system + " " + user, user)

    async def complete_json_array(self, *, system: str, user: str, **kwargs: Any) -> list[Any]:
        return self._infer_array(system + " " + user, user)

    @staticmethod
    def _chunk_ids(user: str) -> list[str]:
        ids = _CHUNK_ID_RE.findall(user or "")
        return ids or ["c-0001"]

    # ---- payload synthesis ----

    def _infer_object(self, hint: str, user: str = "") -> dict[str, Any]:
        # Identity uses a unique uppercase tag in its system prompt:
        # "Schema:" + "- `role`" + "- `mission`". Check on the schema phrase
        # rather than ambient text in chunks.
        if "describing the role this corpus equips" in hint:
            return {
                "role": "RICS Quantity Surveyor",
                "description": "Construction cost professional accredited by RICS.",
                "mission": "Deliver accurate cost advice while complying with RICS rules.",
            }
        hl = hint.lower()
        if "planner" in hl or "outline" in hl or "scenario" in hl:
            return {
                "objective": "Demonstrate adherence to the procedure.",
                "principles": ["PRN-0001"],
                "procedure": "PRC-0001",
                "stop_after_turns": 2,
            }
        if "critic" in hl or "score" in hl:
            return {"score": 0.85, "notes": "Faithful to golden, format OK."}
        if "persona" in hl:
            return {
                "name": "Junior Surveyor",
                "description": "Recent graduate.",
                "expertise_level": "novice",
                "tone": "polite",
            }
        return {"ok": True}

    def _infer_array(self, hint: str, user: str = "") -> list[Any]:
        # Discriminate purely on the extractor's system-prompt tags. The chunks
        # themselves contain words like "procedure" or "principle" so we cannot
        # match on lower-cased prose.
        h = hint
        cids = self._chunk_ids(user)
        cid_a = cids[0]
        cid_b = cids[1] if len(cids) > 1 else cids[0]
        if "PROCEDURE objects" in h:
            return [
                {
                    "name": "Prepare a Bill of Quantities",
                    "purpose": "Measure the works under NRM2.",
                    "steps": [
                        {"order": 1, "action": "Confirm measurement basis (NRM2)."},
                        {"order": 2, "action": "Decompose into work sections."},
                        {"order": 3, "action": "Measure and cite sources."},
                    ],
                    "preconditions": ["Drawings issued"],
                    "postconditions": ["BoQ issued with assumptions log"],
                    "source_chunk_ids": [cid_a],
                },
                {
                    "name": "Issue an Interim Valuation",
                    "purpose": "Value work in place at a valuation date.",
                    "steps": [
                        {"order": 1, "action": "Receive contractor application."},
                        {"order": 2, "action": "Inspect or verify works."},
                        {"order": 3, "action": "Issue Payment Notice."},
                    ],
                    "preconditions": [],
                    "postconditions": [],
                    "source_chunk_ids": [cid_b],
                },
            ]
        if "PRINCIPLE objects" in h:
            return [
                {
                    "statement": "Measure to NRM2 and cite the standard.",
                    "rationale": "Auditability and consistency.",
                    "source_chunk_ids": [cid_a],
                },
                {
                    "statement": "Disclose conflicts of interest proactively.",
                    "rationale": "RICS Rule 3.",
                    "source_chunk_ids": [cid_a],
                },
                {
                    "statement": "Quantify uncertainty with a confidence range.",
                    "rationale": "Risk communication.",
                    "source_chunk_ids": [cid_b],
                },
            ]
        if "KNOWLEDGE objects" in h:
            return [
                {
                    "statement": "NRM2 is the RICS detailed measurement standard.",
                    "citations": [cid_a],
                    "tags": ["nrm"],
                },
                {
                    "statement": "JCT and NEC are dominant UK contract families.",
                    "citations": [cid_b],
                    "tags": ["contracts"],
                },
                {
                    "statement": "A Compensation Event is the NEC change mechanism.",
                    "citations": [cid_b],
                    "tags": ["nec"],
                },
                {
                    "statement": "RICS Rules of Conduct (2022) replaced the older rules.",
                    "citations": [cid_a],
                    "tags": ["ethics"],
                },
            ]
        if "VOCABULARY objects" in h:
            return [
                {
                    "term": "BoQ",
                    "definition": "Bill of Quantities",
                    "aliases": ["bill of quantities"],
                    "source_chunk_ids": [cid_a],
                },
                {
                    "term": "Prelims",
                    "definition": "Preliminaries — site setup costs",
                    "aliases": [],
                    "source_chunk_ids": [cid_a],
                },
                {
                    "term": "WIP",
                    "definition": "Work in place",
                    "aliases": [],
                    "source_chunk_ids": [cid_b],
                },
            ]
        if "ANTI-PATTERN objects" in h:
            return [
                {
                    "description": "Reporting a single-number estimate without uncertainty.",
                    "why_bad": "Misleads the client about risk.",
                    "correction": "Express estimates with a confidence range.",
                    "source_chunk_ids": [cid_a],
                }
            ]
        hl = h.lower()
        if "persona" in hl:
            return [
                {
                    "name": f"Persona {i}",
                    "description": "Stakeholder",
                    "expertise_level": "intermediate",
                    "tone": "neutral",
                }
                for i in range(8)
            ]
        if "scenario" in hl:
            return [
                {
                    "procedure_id": "PRC-0001",
                    "principle_ids": ["PRN-0001"],
                    "persona_id": "PER-0001",
                    "difficulty": "medium",
                    "prompt_seed": "Walk me through a BoQ for a small commercial fit-out.",
                }
            ]
        return []


@pytest.mark.asyncio
async def test_end_to_end_with_fake_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ingestion → curation → sft → rewards → report all wired to real impls."""
    # 1. Use the example BoK fixture as input.
    bok = Path(__file__).resolve().parent.parent / "examples" / "bok_qs"
    assert bok.exists()

    cfg = GyroscopeConfig(
        input_paths=[bok],
        output_dir=tmp_path / "run",
        api_key="test-not-used",
    )
    cfg.sft.n_trajectories = 4
    cfg.sft.n_personas = 2
    cfg.sft.difficulty_mix = {"easy": 1.0}
    cfg.sft.eval_holdout_fraction = 0.5  # ensure both splits non-empty even with tiny counts
    cfg.sft.critic_min_score = 0.0  # don't drop anything from the fake critic
    cfg.sft.dedup_threshold = 1.0  # disable dedup so all trajectories survive
    cfg.curation.max_principles = 10
    cfg.curation.max_procedures = 5
    cfg.curation.max_knowledge_items = 10
    cfg.rewards.reward_budget = 6
    cfg.llm.max_concurrent = 4

    monkeypatch.setattr("gyroscope.runner.LLMClient", FakeLLM)

    from gyroscope.runner import AutonomousRunner

    runner = AutonomousRunner(cfg, threshold=0.0, max_iterations=0)
    art = await runner.run()

    # Artefacts exist on disk
    run = cfg.output_dir
    assert (run / "documents.jsonl").exists()
    assert (run / "golden.md").exists()
    assert (run / "golden.json").exists()
    assert (run / "sft.jsonl").exists()
    assert (run / "eval.jsonl").exists()
    assert (run / "rewards" / "rewards.py").exists()
    assert (run / "rewards" / "reward_spec.yaml").exists()
    assert (run / "rewards" / "_lib.py").exists()
    assert (run / "report.md").exists()
    assert (run / "report.html").exists()
    assert (run / "report.json").exists()
    assert (run / "history.json").exists()

    # Golden round-trips
    golden = GoldenDocument.model_validate_json((run / "golden.json").read_text())
    assert golden.identity.role
    assert golden.principles
    assert golden.procedures

    # Generated rewards package is importable and callable. The codegen
    # writes a `rewards/` *package* with relative imports, so insert the
    # parent dir on sys.path and import the package by name.
    sys.path.insert(0, str(run))
    try:
        if "rewards" in sys.modules:
            del sys.modules["rewards"]
        import rewards as gen_rewards  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    assert hasattr(gen_rewards, "REWARDS")
    assert isinstance(gen_rewards.REWARDS, list)
    assert gen_rewards.REWARDS, "generated REWARDS list is empty"
    for fn in gen_rewards.REWARDS:
        out = fn(
            prompts=["What is a BoQ?"],
            completions=["A Bill of Quantities is a priced schedule. [KNW-0001]"],
        )
        assert isinstance(out, list) and len(out) == 1
        assert 0.0 <= float(out[0]) <= 1.0

    # Train and eval splits don't leak procedures
    from gyroscope.eval.pipeline import leakage_check

    assert leakage_check(art.train, art.eval) == set()


@pytest.mark.asyncio
async def test_quality_report_assembles_from_real_artefacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a full run, the quality report metrics produce sensible numbers."""
    bok = Path(__file__).resolve().parent.parent / "examples" / "bok_qs"
    cfg = GyroscopeConfig(input_paths=[bok], output_dir=tmp_path / "run", api_key="test")
    cfg.sft.n_trajectories = 4
    cfg.sft.n_personas = 2
    cfg.sft.difficulty_mix = {"easy": 1.0}
    cfg.sft.eval_holdout_fraction = 0.5
    cfg.sft.critic_min_score = 0.0
    cfg.sft.dedup_threshold = 1.0
    cfg.rewards.reward_budget = 6

    monkeypatch.setattr("gyroscope.runner.LLMClient", FakeLLM)

    from gyroscope.runner import AutonomousRunner

    runner = AutonomousRunner(cfg, threshold=0.0, max_iterations=0)
    art = await runner.run()

    report = assemble_report(
        golden=art.golden,
        source_texts=[d.text for d in art.documents],
        trajectories=art.train,
        reward_specs=art.rewards,
    )
    d = report.to_dict()
    assert set(d["axes"].keys()) == {
        "coverage",
        "faithfulness",
        "diversity",
        "trainability",
        "reward_soundness",
    }
    # Trainability should be high — the fake LLM produces well-formed turns.
    assert report.axes["trainability"].score >= 80
