"""Tests for the dict-based ``RENDERERS`` dispatch in the reward codegen.

The codegen used to dispatch on :class:`RewardKind` with a long ``if/elif``
chain that no plugin could extend. The refactor exposes a module-level
:data:`RENDERERS` mapping plus :func:`register_renderer`, so a plugin can
swap in a custom body for an existing kind (or, in principle, install a
brand-new kind via ``RewardKind``-extension) without monkeypatching module
internals.
"""

from __future__ import annotations

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
from gyroscope.rewards import codegen as codegen_mod
from gyroscope.rewards.codegen import (
    RENDERERS,
    emit_rewards_module,
    register_renderer,
)
from gyroscope.rewards.spec import RewardBundle


def _tiny_golden() -> GoldenDocument:
    return GoldenDocument(
        identity=Identity(role="Test", description="d", mission="m"),
        principles=[Principle(id="PRN-0001", statement="be precise")],
        procedures=[
            Procedure(
                id="PRC-0001",
                name="workflow",
                purpose="do",
                steps=[ProcedureStep(order=1, action="go")],
            )
        ],
        knowledge=[KnowledgeItem(id="KNW-0001", statement="fact")],
        vocabulary=[VocabularyTerm(term="alpha", definition="d")],
        anti_patterns=[AntiPattern(id="ANT-0001", description="bad", why_bad="r", correction="c")],
    )


def test_renderers_dict_covers_all_built_in_kinds() -> None:
    """Every :class:`RewardKind` must have a registered renderer at module
    scope — without this the lookup in :func:`_render_function` silently
    falls back to a no-op body for built-in kinds."""
    for kind in RewardKind:
        assert kind in RENDERERS, f"missing renderer for {kind!r}"


def test_register_renderer_overrides_built_in_format_kind(
    tmp_path: Path,
) -> None:
    """A plugin can replace an existing renderer via ``register_renderer``
    and the override must show up in the emitted ``rewards.py``."""
    original = RENDERERS[RewardKind.FORMAT]
    sentinel = "    return [0.987 for _ in completions]\n"

    def custom_format_body(spec: RewardSpec) -> str:
        # Body intentionally ignores the spec — we only need to prove the
        # override is wired into the codegen path.
        return sentinel

    register_renderer(RewardKind.FORMAT, custom_format_body)
    try:
        bundle = RewardBundle(
            specs=[
                RewardSpec(
                    name="custom_format",
                    kind=RewardKind.FORMAT,
                    description="overridden via registry",
                    weight=1.0,
                    config={"sections": ["Summary"]},
                )
            ],
            golden_role="Test",
            version="1",
        )
        rewards_path = emit_rewards_module(bundle, _tiny_golden(), tmp_path)
        text = rewards_path.read_text(encoding="utf-8")
        # The override's sentinel body must appear verbatim in the emitted
        # function — proof the dispatch table was consulted, not the old
        # if/elif chain (which would have rendered ``format_sections``).
        assert "return [0.987 for _ in completions]" in text
        assert "format_sections" not in text
    finally:
        # Always restore the built-in renderer so later tests are unaffected.
        register_renderer(RewardKind.FORMAT, original)


def test_render_function_lookup_uses_module_renderers_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching ``RENDERERS[kind]`` (via monkeypatch.setitem) flows through
    the :func:`_render_function` lookup. This pins the contract that the
    dispatch is a dict lookup, not a frozen closure captured at import."""
    sentinel = "    return [0.111 for _ in completions]\n"

    def fake(spec: RewardSpec) -> str:
        return sentinel

    monkeypatch.setitem(codegen_mod.RENDERERS, RewardKind.LENGTH, fake)

    spec = RewardSpec(
        name="length_demo",
        kind=RewardKind.LENGTH,
        description="d",
        weight=1.0,
        config={"min_tokens": 1, "max_tokens": 2, "sweet_spot": 1},
    )
    rendered = codegen_mod._render_function(spec)
    assert "return [0.111 for _ in completions]" in rendered
