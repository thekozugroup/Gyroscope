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
