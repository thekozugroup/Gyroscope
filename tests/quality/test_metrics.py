"""Tests for the deterministic quality metrics used by the critique loop."""

from __future__ import annotations

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
    Trajectory,
    TrajectoryMessage,
)
from gyroscope.quality.metrics import (
    assemble_report,
    coverage_score,
    diversity_score,
    faithfulness_score,
    reward_soundness_score,
    trainability_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _golden(
    n_principles: int = 30, n_procedures: int = 15, n_knowledge: int = 100
) -> GoldenDocument:
    return GoldenDocument(
        identity=Identity(role="Surveyor", description="d", mission="m"),
        principles=[
            Principle(
                id=f"PRN-{i:04d}",
                statement=f"Principle {i}",
                source_chunk_ids=["c1"],
            )
            for i in range(1, n_principles + 1)
        ],
        procedures=[
            Procedure(
                id=f"PRC-{i:04d}",
                name=f"Procedure {i}",
                purpose="p",
                steps=[ProcedureStep(order=1, action="measure")],
            )
            for i in range(1, n_procedures + 1)
        ],
        knowledge=[
            KnowledgeItem(id=f"KNW-{i:04d}", statement=f"Fact {i}", citations=["c1"])
            for i in range(1, n_knowledge + 1)
        ],
        anti_patterns=[
            AntiPattern(
                id="ANT-0001",
                description="bad",
                why_bad="reasons",
                correction="do this",
            )
        ],
    )


def _traj(idx: int, procedure_id: str, user: str, assistant: str) -> Trajectory:
    return Trajectory(
        id=f"TRJ-{idx:04d}",
        scenario_id=f"SCN-{idx:04d}",
        system="You are a Surveyor.",
        messages=[
            TrajectoryMessage(role="user", content=user),
            TrajectoryMessage(role="assistant", content=assistant),
        ],
        tags={"procedure_ids": [procedure_id], "persona": "p1", "difficulty": "medium"},
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_scores_high_when_at_targets():
    g = _golden(30, 15, 100)
    source_texts = [g.to_markdown()]
    a = coverage_score(g, source_texts)
    assert a.score >= 80


def test_coverage_scores_low_when_sparse():
    g = _golden(1, 1, 1)
    a = coverage_score(g, ["lots of unrelated source vocabulary that the golden never mentions"])
    assert a.score < 50


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def test_faithfulness_high_when_all_cited():
    g = _golden()
    a = faithfulness_score(g, trajectories=[])
    assert a.score >= 90


def test_faithfulness_drops_when_citations_missing():
    g = GoldenDocument(
        identity=Identity(role="X", description="d", mission="m"),
        principles=[Principle(id="PRN-0001", statement="No cite", source_chunk_ids=[])],
        knowledge=[KnowledgeItem(id="KNW-0001", statement="No cite", citations=[])],
    )
    a = faithfulness_score(g, trajectories=[])
    assert a.score < 50


def test_faithfulness_flags_weak_trajectory_overlap():
    g = _golden(3, 3, 3)
    bad = _traj(1, "PRC-0001", "user q", "totally unrelated lorem ipsum dolor sit amet")
    a = faithfulness_score(g, trajectories=[bad])
    assert "weak golden overlap" in " ".join(a.notes)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------


def test_diversity_low_when_all_trajectories_identical():
    trajs = [
        _traj(i, "PRC-0001", "same q", "same a same a same a same a same a") for i in range(10)
    ]
    a = diversity_score(trajs)
    assert a.score < 40


def test_diversity_high_with_varied_trajectories():
    trajs = [
        _traj(
            i, f"PRC-{i:04d}", f"unique question {i}", f"unique answer {i} alpha beta gamma delta"
        )
        for i in range(10)
    ]
    a = diversity_score(trajs)
    assert a.score > 60


# ---------------------------------------------------------------------------
# Trainability
# ---------------------------------------------------------------------------


def test_trainability_full_when_clean():
    trajs = [_traj(i, "PRC-0001", "q", "a") for i in range(5)]
    a = trainability_score(trajs)
    assert a.score == 100


def test_trainability_penalises_empty_or_collapsed_turns():
    bad = Trajectory(
        id="TRJ-0001",
        scenario_id="SCN-0001",
        system="",
        messages=[TrajectoryMessage(role="user", content="")],
    )
    a = trainability_score([bad])
    assert a.score == 0


# ---------------------------------------------------------------------------
# Reward soundness
# ---------------------------------------------------------------------------


def _spec(name: str, kind: RewardKind, **kwargs) -> RewardSpec:
    return RewardSpec(name=name, kind=kind, description="x", **kwargs)


def test_reward_soundness_high_with_good_mix():
    specs = [
        _spec("reward_safety", RewardKind.SAFETY, config={"anti_pattern_ids": ["ANT-0001"]}),
        _spec("reward_principle_1", RewardKind.PRINCIPLE, principle_ids=["PRN-0001"]),
        _spec("reward_procedure_1", RewardKind.PROCEDURE, procedure_ids=["PRC-0001"]),
        _spec("reward_format_sec", RewardKind.FORMAT, config={"sections": ["Findings"]}),
    ]
    g = _golden(1, 1, 1)
    a = reward_soundness_score(specs, golden=g)
    assert a.score >= 80


def test_reward_soundness_low_when_missing_kinds():
    specs = [_spec("reward_format", RewardKind.FORMAT, config={"sections": ["X"]})]
    a = reward_soundness_score(specs)
    assert a.score < 50


def test_reward_soundness_flags_broken_refs():
    g = _golden(1, 1, 1)
    specs = [
        _spec("reward_principle", RewardKind.PRINCIPLE, principle_ids=["PRN-9999"]),
        _spec("reward_procedure", RewardKind.PROCEDURE, procedure_ids=["PRC-9999"]),
        _spec("reward_safety", RewardKind.SAFETY),
    ]
    a = reward_soundness_score(specs, golden=g)
    assert "broken refs" in " ".join(a.notes)


def test_reward_soundness_handles_empty():
    a = reward_soundness_score([])
    assert a.score == 0


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def test_assemble_report_runs_all_axes():
    g = _golden(10, 5, 20)
    trajs = [_traj(i, "PRC-0001", f"q{i}", f"a{i} measure quantities") for i in range(5)]
    specs = [
        _spec("reward_safety", RewardKind.SAFETY),
        _spec("reward_principle", RewardKind.PRINCIPLE, principle_ids=["PRN-0001"]),
        _spec("reward_procedure", RewardKind.PROCEDURE, procedure_ids=["PRC-0001"]),
    ]
    report = assemble_report(
        golden=g, source_texts=[g.to_markdown()], trajectories=trajs, reward_specs=specs
    )
    assert set(report.axes.keys()) == {
        "coverage",
        "faithfulness",
        "diversity",
        "trainability",
        "reward_soundness",
    }
    d = report.to_dict()
    assert "overall" in d
