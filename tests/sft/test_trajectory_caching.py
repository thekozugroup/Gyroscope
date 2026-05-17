"""Prompt-caching guarantees: the assistant + critic system prompt must be
byte-identical across every scenario sharing the same GoldenDocument.

This is the core invariant that lets Anthropic prompt caching reuse the
cached prefix across ~60k LLM calls in a 5k-trajectory run. If the
per-scenario selection ever leaks back into the cached prefix the cache
hit rate collapses and we burn ~5-10x the tokens we should.
"""

from __future__ import annotations

from gyroscope.core.models import Scenario
from gyroscope.sft.trajectory import (
    build_scenario_suffix,
    build_stable_system_prefix,
)

from .conftest import make_golden


def _scenario(
    sid: str,
    *,
    principle_ids: list[str],
    procedure_id: str | None,
    seed: str = "explain things",
) -> Scenario:
    return Scenario(
        id=sid,
        procedure_id=procedure_id,
        principle_ids=principle_ids,
        persona_id="PER-0001",
        difficulty="medium",
        prompt_seed=seed,
    )


def test_stable_prefix_is_identical_across_different_scenarios():
    """Two scenarios with completely different selections must produce the
    same stable prefix — the prefix only depends on ``golden``."""
    golden = make_golden(n_procedures=3, n_principles=4)

    s1 = _scenario(
        "SCN-0001",
        principle_ids=["PRN-0001", "PRN-0002"],
        procedure_id="PRC-0001",
        seed="how do I do thing A",
    )
    s2 = _scenario(
        "SCN-0002",
        principle_ids=["PRN-0003"],
        procedure_id="PRC-0003",
        seed="how do I do thing B",
    )

    prefix1 = build_stable_system_prefix(golden)
    prefix2 = build_stable_system_prefix(golden)
    assert prefix1 == prefix2, "prefix must not depend on call site"

    # Even though the scenarios point at different principles and procedures,
    # the stable prefix that goes through the cache must be byte-identical.
    suffix1 = build_scenario_suffix(s1, golden)
    suffix2 = build_scenario_suffix(s2, golden)
    assert suffix1 != suffix2, "per-scenario suffix should differ across scenarios"

    # And crucially, the per-scenario selections do NOT appear in the cached
    # prefix in a way that would make the prefix vary by scenario.
    assert prefix1 == build_stable_system_prefix(golden)


def test_stable_prefix_contains_full_golden_content():
    """The prefix must encode the FULL principle / procedure list so the
    cached prefix is the heavy chunk; the per-scenario suffix can stay
    lightweight."""
    golden = make_golden(n_procedures=3, n_principles=4)
    prefix = build_stable_system_prefix(golden)
    for p in golden.principles:
        assert p.id in prefix, f"principle {p.id} missing from stable prefix"
    for proc in golden.procedures:
        assert proc.id in prefix, f"procedure {proc.id} missing from stable prefix"


def test_scenario_suffix_lists_selected_ids():
    """The per-scenario suffix should advertise the selected principles /
    procedure so the assistant knows what to focus on this turn."""
    golden = make_golden(n_procedures=2, n_principles=3)
    s = _scenario(
        "SCN-0010",
        principle_ids=["PRN-0002"],
        procedure_id="PRC-0001",
    )
    suffix = build_scenario_suffix(s, golden)
    assert "PRN-0002" in suffix
    assert "PRC-0001" in suffix


def test_prefix_cache_does_not_leak_across_distinct_goldens() -> None:
    """Regression test for the id()-keyed cache bug found in round 8.

    Builds golden A, primes the cache, drops the reference, garbage-collects,
    then builds a different golden B and asserts B's prefix carries B's ids
    rather than A's — even if CPython happens to recycle A's id() for B.
    """
    import gc

    from gyroscope.sft.trajectory import build_stable_system_prefix

    from .conftest import make_golden

    a = make_golden()
    # Tag A with a unique principle id we can search for.
    a.principles[0].id = "PRN-AAAA"
    a.principles[0].statement = "principle A unique marker"
    prefix_a = build_stable_system_prefix(a)
    assert "PRN-AAAA" in prefix_a

    a_id = id(a)
    del a
    gc.collect()

    # Build B with a different unique marker. We do NOT control whether
    # Python reuses the previous id, but the content-keyed cache must not
    # confuse B for A regardless of that choice.
    b = make_golden()
    b.principles[0].id = "PRN-BBBB"
    b.principles[0].statement = "principle B unique marker"

    prefix_b = build_stable_system_prefix(b)
    assert "PRN-BBBB" in prefix_b, (
        f"prefix B leaked stale content from A (id reuse: {id(b) == a_id}); "
        f"got prefix that contains PRN-AAAA={'PRN-AAAA' in prefix_b!r}"
    )
    assert "PRN-AAAA" not in prefix_b, "prefix B contains A's principle id"
    assert prefix_a != prefix_b


def test_prefix_cache_collapses_identical_distinct_instances() -> None:
    """Two distinct GoldenDocument instances with the SAME content must
    share the same memoized prefix string object — that's the upside of
    content-keying.
    """
    from gyroscope.sft.trajectory import build_stable_system_prefix

    from .conftest import make_golden

    a = make_golden()
    b = make_golden()
    assert a is not b

    pa = build_stable_system_prefix(a)
    pb = build_stable_system_prefix(b)

    # Same content → same memoized string object (cache hit on second call).
    assert pa is pb, "content-keyed cache should hand out the same string for equal content"


def test_prefix_cache_invalidates_on_procedure_step_mutation() -> None:
    """If a procedure's steps change (not its id/name), the cached prefix
    must be re-derived. Round-9 widened the fingerprint to cover this."""
    from gyroscope.core.models import ProcedureStep
    from gyroscope.sft.trajectory import build_stable_system_prefix

    from .conftest import make_golden

    g = make_golden()
    # Replace the first procedure's step list with a unique-marker step.
    proc = g.procedures[0]
    proc.steps = [ProcedureStep(order=1, action="STEP-ORIGINAL unique marker")]
    p1 = build_stable_system_prefix(g)
    assert "STEP-ORIGINAL unique marker" in p1

    # Mutate the step body in place — id/name unchanged.
    proc.steps = [ProcedureStep(order=1, action="STEP-MUTATED other marker")]
    p2 = build_stable_system_prefix(g)
    assert "STEP-MUTATED other marker" in p2
    assert "STEP-ORIGINAL unique marker" not in p2
    assert p1 != p2


def test_prefix_cache_invalidates_on_vocab_definition_mutation() -> None:
    """If a vocabulary term's definition changes, the cached prefix must
    be re-derived even though the term itself is unchanged."""
    from gyroscope.core.models import VocabularyTerm
    from gyroscope.sft.trajectory import build_stable_system_prefix

    from .conftest import make_golden

    g = make_golden()
    g.vocabulary = [VocabularyTerm(term="X", definition="DEF-ORIGINAL")]
    p1 = build_stable_system_prefix(g)
    assert "DEF-ORIGINAL" in p1

    g.vocabulary = [VocabularyTerm(term="X", definition="DEF-MUTATED")]
    p2 = build_stable_system_prefix(g)
    assert "DEF-MUTATED" in p2
    assert "DEF-ORIGINAL" not in p2


def test_prefix_cache_invalidates_on_anti_pattern_body_mutation() -> None:
    """If an anti-pattern's description or correction changes, the cached
    prefix must be re-derived even though the id is unchanged."""
    from gyroscope.core.models import AntiPattern
    from gyroscope.sft.trajectory import build_stable_system_prefix

    from .conftest import make_golden

    g = make_golden()
    g.anti_patterns = [
        AntiPattern(
            id="ANT-0001",
            description="ANTI-ORIGINAL desc",
            why_bad="r",
            correction="CORR-ORIGINAL fix",
        )
    ]
    p1 = build_stable_system_prefix(g)
    assert "ANTI-ORIGINAL desc" in p1
    assert "CORR-ORIGINAL fix" in p1

    g.anti_patterns = [
        AntiPattern(
            id="ANT-0001",
            description="ANTI-MUTATED desc",
            why_bad="r",
            correction="CORR-MUTATED fix",
        )
    ]
    p2 = build_stable_system_prefix(g)
    assert "ANTI-MUTATED desc" in p2
    assert "ANTI-ORIGINAL desc" not in p2


def test_stream_swarm_primes_prefix_cache_exactly_once(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """`stream_swarm` must call `build_stable_system_prefix` before workers
    start so they all hit the cache rather than racing to build the prefix.
    """
    import asyncio

    from gyroscope.core.config import SFTConfig
    from gyroscope.sft import swarm as swarm_mod
    from gyroscope.sft import trajectory as trajectory_mod

    from .conftest import make_golden

    g = make_golden()
    cfg = SFTConfig(n_trajectories=2, n_personas=1, difficulty_mix={"easy": 1.0})

    call_count = {"n": 0}
    real = trajectory_mod.build_stable_system_prefix

    def counting(golden):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        return real(golden)

    # `stream_swarm` imports build_stable_system_prefix locally, so patch
    # the source module — the local import resolves to the patched object.
    monkeypatch.setattr(trajectory_mod, "build_stable_system_prefix", counting)

    from gyroscope.core.models import Persona, Scenario

    async def _one_persona(*args, **kwargs):  # type: ignore[no-untyped-def]
        return [Persona(id="PER-0001", name="P", description="d")]

    async def _two_scenarios(*args, **kwargs):  # type: ignore[no-untyped-def]
        return [
            Scenario(
                id=f"SCN-{i:04d}",
                procedure_id=g.procedures[0].id,
                principle_ids=[],
                persona_id="PER-0001",
                difficulty="easy",
                prompt_seed=f"q {i}",
            )
            for i in range(2)
        ]

    # Stub _build_one so workers don't actually call build_trajectory (which
    # would call build_stable_system_prefix internally and skew the count).
    async def _stub_build_one(scenario, *args, **kwargs):  # type: ignore[no-untyped-def]
        from gyroscope.core.models import Trajectory, TrajectoryMessage

        return Trajectory(
            id=f"TRJ-{scenario.id[-4:]}",
            scenario_id=scenario.id,
            system="s",
            messages=[
                TrajectoryMessage(role="user", content="u"),
                TrajectoryMessage(role="assistant", content="a"),
            ],
            tags={"procedure_ids": [scenario.procedure_id], "difficulty": "easy"},
            quality_score=0.9,
        )

    monkeypatch.setattr(swarm_mod, "generate_personas", _one_persona)
    monkeypatch.setattr(swarm_mod, "generate_scenarios", _two_scenarios)
    monkeypatch.setattr(swarm_mod, "_build_one", _stub_build_one)

    class _FakeLLM:
        @property
        def config(self):  # type: ignore[no-untyped-def]
            class _C:
                class _L:
                    max_concurrent = 2

                llm = _L()

            return _C()

    async def drain():  # type: ignore[no-untyped-def]
        async for _ in swarm_mod.stream_swarm(g, _FakeLLM(), cfg):
            pass

    asyncio.run(drain())

    # The priming call must happen exactly once — workers never have to
    # rebuild the prefix because _build_one is stubbed and would not call
    # build_stable_system_prefix anyway. Two scenarios in flight, one
    # priming call asserted: cache hit ratio is 100%.
    assert call_count["n"] == 1, (
        f"stream_swarm should prime the prefix exactly once; got {call_count['n']}"
    )
