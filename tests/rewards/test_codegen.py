"""Tests for the codegen module: generated module must import and run cleanly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from gyroscope.core.models import (
    AntiPattern,
    GoldenDocument,
    Identity,
    KnowledgeItem,
    Principle,
    Procedure,
    ProcedureStep,
    RewardKind,
    RewardSpec,
    VocabularyTerm,
)
from gyroscope.rewards.codegen import emit_rewards_module
from gyroscope.rewards.spec import RewardBundle


def _bundle_with_all_kinds() -> RewardBundle:
    specs: list[RewardSpec] = [
        RewardSpec(
            name="format_sections",
            kind=RewardKind.FORMAT,
            description="Sections present",
            weight=0.5,
            config={"sections": ["Summary", "Answer"]},
        ),
        RewardSpec(
            name="lexical_terms",
            kind=RewardKind.LEXICAL,
            description="Use domain terms",
            weight=0.3,
            config={"required": ["alpha"], "forbidden": ["banned"], "case_sensitive": False},
        ),
        RewardSpec(
            name="principle_main",
            kind=RewardKind.PRINCIPLE,
            description="Follow the main principle",
            weight=1.0,
            principle_ids=["PRN-0001"],
            config={
                "principle_id": "PRN-0001",
                "principle_statement": "Always be helpful and precise.",
                "judge_model": "claude-test",
            },
        ),
        RewardSpec(
            name="procedure_workflow",
            kind=RewardKind.PROCEDURE,
            description="Follow workflow",
            weight=1.0,
            procedure_ids=["PRC-0001"],
            config={"procedure_id": "PRC-0001", "ordered": True, "ordered_steps": ["gather", "design", "ship"]},
        ),
        RewardSpec(
            name="safety_main",
            kind=RewardKind.SAFETY,
            description="Avoid anti-patterns",
            weight=2.0,
            config={"anti_pattern_ids": ["ANT-0001"], "anti_pattern_terms": [["bad", "stuff"]]},
        ),
        RewardSpec(
            name="citation_knw",
            kind=RewardKind.CITATION,
            description="Cite KNW ids",
            weight=1.0,
            config={"require_ids": True, "min_citations": 1},
        ),
        RewardSpec(
            name="length_budget",
            kind=RewardKind.LENGTH,
            description="Stay in budget",
            weight=0.25,
            config={"min_tokens": 10, "max_tokens": 500, "sweet_spot": 100},
        ),
    ]
    return RewardBundle(specs=specs, golden_role="Test Role", version="1")


def _tiny_golden() -> GoldenDocument:
    return GoldenDocument(
        identity=Identity(role="Test Role", description="d", mission="m"),
        principles=[Principle(id="PRN-0001", statement="be precise")],
        procedures=[
            Procedure(
                id="PRC-0001",
                name="workflow",
                purpose="do the work",
                steps=[
                    ProcedureStep(order=1, action="gather"),
                    ProcedureStep(order=2, action="design"),
                    ProcedureStep(order=3, action="ship"),
                ],
            )
        ],
        knowledge=[KnowledgeItem(id="KNW-0001", statement="fact")],
        vocabulary=[VocabularyTerm(term="alpha", definition="d")],
        anti_patterns=[
            AntiPattern(
                id="ANT-0001",
                description="bad stuff happens",
                why_bad="risky",
                correction="do not",
            )
        ],
    )


def _load_generated(path: Path, mod_name: str = "gen_rewards_pkg") -> object:
    """Import the generated package from disk under a unique name."""
    pkg_dir = path.parent
    # We import the package by adding the parent of `rewards/` to sys.path.
    parent = pkg_dir.parent
    sys.path.insert(0, str(parent))
    try:
        # Clean stale imports.
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


def test_codegen_emits_expected_files(tmp_path: Path) -> None:
    bundle = _bundle_with_all_kinds()
    golden = _tiny_golden()
    rewards_path = emit_rewards_module(bundle, golden, tmp_path)
    pkg = rewards_path.parent
    assert rewards_path.exists()
    assert (pkg / "__init__.py").exists()
    assert (pkg / "_lib.py").exists()
    assert (pkg / "reward_spec.yaml").exists()
    # File names are correct
    assert rewards_path.name == "rewards.py"


def test_generated_module_is_importable_and_callable(tmp_path: Path) -> None:
    bundle = _bundle_with_all_kinds()
    golden = _tiny_golden()
    rewards_path = emit_rewards_module(bundle, golden, tmp_path)

    module = _load_generated(rewards_path)
    assert hasattr(module, "REWARDS")
    assert hasattr(module, "WEIGHTS")
    rewards = module.REWARDS  # type: ignore[attr-defined]
    weights = module.WEIGHTS  # type: ignore[attr-defined]

    assert isinstance(rewards, list)
    assert len(rewards) == len(bundle.specs)
    for fn in rewards:
        assert callable(fn)

    # Names line up with WEIGHTS mapping.
    spec_names = {s.name for s in bundle.specs}
    assert set(weights.keys()) == spec_names

    sample_prompts = ["What now?", "Other question"]
    sample_completions = [
        (
            "# Summary\nUse alpha when needed.\n# Answer\n"
            "First gather, then design, then ship. [KNW-0001]"
        ),
        "no structure at all just plain text",
    ]
    for fn in rewards:
        out = fn(sample_prompts, sample_completions)
        assert isinstance(out, list)
        assert len(out) == len(sample_completions)
        for v in out:
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0


def test_generated_module_only_depends_on_stdlib_and_jsonschema(tmp_path: Path) -> None:
    bundle = _bundle_with_all_kinds()
    golden = _tiny_golden()
    rewards_path = emit_rewards_module(bundle, golden, tmp_path)
    text = rewards_path.read_text(encoding="utf-8")
    # Generated rewards.py should never import gyroscope itself.
    assert "gyroscope" not in text


def test_generated_lib_has_no_gyroscope_imports(tmp_path: Path) -> None:
    bundle = _bundle_with_all_kinds()
    golden = _tiny_golden()
    rewards_path = emit_rewards_module(bundle, golden, tmp_path)
    lib_text = (rewards_path.parent / "_lib.py").read_text(encoding="utf-8")
    assert "from gyroscope" not in lib_text
    assert "import gyroscope" not in lib_text


def test_principle_reward_accepts_injected_judge(tmp_path: Path) -> None:
    bundle = _bundle_with_all_kinds()
    golden = _tiny_golden()
    rewards_path = emit_rewards_module(bundle, golden, tmp_path)
    module = _load_generated(rewards_path)

    principle_fn = next(
        fn for fn in module.REWARDS  # type: ignore[attr-defined]
        if fn.__name__ == "reward_principle_main"
    )

    calls: list[str] = []

    def fake_judge(prompts, completions, criterion):
        calls.append(criterion)
        return [0.33 for _ in completions]

    scores = principle_fn(["p"], ["c"], judge=fake_judge)
    assert scores == [pytest.approx(0.33)]
    assert calls == ["Always be helpful and precise."]


def test_reward_spec_yaml_roundtrip(tmp_path: Path) -> None:
    bundle = _bundle_with_all_kinds()
    golden = _tiny_golden()
    rewards_path = emit_rewards_module(bundle, golden, tmp_path)
    loaded = RewardBundle.from_yaml(rewards_path.parent / "reward_spec.yaml")
    assert loaded.golden_role == bundle.golden_role
    assert [s.name for s in loaded.specs] == [s.name for s in bundle.specs]
