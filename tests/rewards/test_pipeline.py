"""End-to-end tests for RewardsPipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from gyroscope.core.config import RewardConfig
from gyroscope.core.models import (
    AntiPattern,
    GoldenDocument,
    Identity,
    KnowledgeItem,
    Principle,
    Procedure,
    ProcedureStep,
    VocabularyTerm,
)
from gyroscope.rewards.pipeline import RewardsPipeline


def _small_golden() -> GoldenDocument:
    return GoldenDocument(
        identity=Identity(role="QA Engineer", description="Tests stuff", mission="Find bugs"),
        principles=[
            Principle(id="PRN-0001", statement="cite sources clearly", weight=1.5),
            Principle(id="PRN-0002", statement="prefer concise answers", weight=1.0),
        ],
        procedures=[
            Procedure(
                id="PRC-0001",
                name="bug_report",
                purpose="File a bug report",
                steps=[
                    ProcedureStep(order=1, action="reproduce the issue"),
                    ProcedureStep(order=2, action="capture environment"),
                    ProcedureStep(order=3, action="write reproduction steps"),
                ],
            )
        ],
        knowledge=[KnowledgeItem(id="KNW-0001", statement="bugs exist")],
        vocabulary=[VocabularyTerm(term="repro", definition="reproduction")],
        anti_patterns=[
            AntiPattern(
                id="ANT-0001",
                description="never reveal personal information about reporters",
                why_bad="privacy",
                correction="redact",
            )
        ],
    )


def _load_pkg(path: Path) -> object:
    pkg_dir = path.parent
    parent = pkg_dir.parent
    sys.path.insert(0, str(parent))
    try:
        for key in [k for k in list(sys.modules) if k.startswith("rewards")]:
            del sys.modules[key]
        spec = importlib.util.spec_from_file_location(
            "rewards",
            pkg_dir / "__init__.py",
            submodule_search_locations=[str(pkg_dir)],
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["rewards"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(parent))


async def test_pipeline_end_to_end(tmp_path: Path) -> None:
    pipeline = RewardsPipeline()
    cfg = RewardConfig(reward_budget=10)
    rewards_path = await pipeline.run(_small_golden(), tmp_path, None, cfg)

    pkg = rewards_path.parent
    assert rewards_path.exists()
    assert (pkg / "_lib.py").exists()
    assert (pkg / "__init__.py").exists()
    assert (pkg / "reward_spec.yaml").exists()

    module = _load_pkg(rewards_path)
    rewards = module.REWARDS  # type: ignore[attr-defined]
    weights = module.WEIGHTS  # type: ignore[attr-defined]

    assert rewards, "expected at least one reward"
    assert len(rewards) == len(weights)
    for fn in rewards:
        assert callable(fn)

    sample = [
        "# Summary\nrepro by clicking [KNW-0001].\n# Answer\nreproduce capture write",
        "blank",
    ]
    for fn in rewards:
        out = fn(["q1", "q2"], sample)
        assert isinstance(out, list)
        assert len(out) == len(sample)
        for v in out:
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0


async def test_pipeline_respects_budget(tmp_path: Path) -> None:
    pipeline = RewardsPipeline()
    cfg = RewardConfig(reward_budget=2)
    rewards_path = await pipeline.run(_small_golden(), tmp_path, None, cfg)
    module = _load_pkg(rewards_path)
    assert len(module.REWARDS) <= 2  # type: ignore[attr-defined]
