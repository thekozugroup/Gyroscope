"""Runtime reward primitives.

Each primitive in this module accepts ``completions: list[str]`` plus its own
keyword-only configuration and returns ``list[float]`` of the same length, with
every value in the closed interval ``[0.0, 1.0]``.

The primitives are:

- pure-Python and framework-agnostic;
- deterministic where possible (judge-based scoring is the only exception);
- safe — they catch their own per-row errors and emit ``0.0`` for malformed
  completions rather than raising.

They are imported both by :mod:`gyroscope.rewards.codegen` (so generated
``rewards.py`` modules can call them through a self-contained copy) and by the
test-suite (so we never have to load the generated module to validate them).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

try:  # ``jsonschema`` is a runtime dependency, but soft-import for robustness.
    import jsonschema
except Exception:  # pragma: no cover - exercised only when dep is missing
    jsonschema = None

logger = logging.getLogger(__name__)

__all__ = [
    "citation",
    "format_json_schema",
    "format_regex",
    "format_sections",
    "length",
    "lexical",
    "principle_judge",
    "procedure_check",
    "safety",
    "tokenize",
]


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric token list. Deliberately tokenizer-free."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _clip(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _safe_for_each(
    completions: Sequence[str],
    fn: Callable[[str], float],
) -> list[float]:
    """Apply ``fn`` to every completion; per-row exceptions become 0.0."""
    out: list[float] = []
    for idx, c in enumerate(completions):
        if not isinstance(c, str):
            logger.debug("non-string completion at index %d -> 0.0", idx)
            out.append(0.0)
            continue
        try:
            out.append(_clip(fn(c)))
        except Exception:
            logger.exception("reward primitive crashed on row %d; returning 0.0", idx)
            out.append(0.0)
    return out


# ---------------------------------------------------------------------------
# Format primitives
# ---------------------------------------------------------------------------


def format_regex(completions: Sequence[str], *, pattern: str) -> list[float]:
    """Reward 1.0 if the completion matches ``pattern`` (``re.search``), else 0.0."""
    try:
        compiled = re.compile(pattern, re.DOTALL | re.MULTILINE)
    except re.error:
        logger.exception("format_regex: invalid pattern %r", pattern)
        return [0.0 for _ in completions]

    def score(c: str) -> float:
        return 1.0 if compiled.search(c) else 0.0

    return _safe_for_each(completions, score)


def format_json_schema(
    completions: Sequence[str],
    *,
    schema: dict[str, Any],
) -> list[float]:
    """Reward 1.0 if the completion parses as JSON validating against ``schema``."""

    def score(c: str) -> float:
        import json

        try:
            payload = json.loads(c)
        except Exception:
            return 0.0
        if jsonschema is None:  # pragma: no cover - dep present in CI
            return 1.0  # parsed-only fallback
        try:
            jsonschema.validate(payload, schema)
        except Exception:
            return 0.0
        return 1.0

    return _safe_for_each(completions, score)


def format_sections(completions: Sequence[str], *, sections: list[str]) -> list[float]:
    """Fraction of required Markdown headers that appear in each completion.

    A "section" can be supplied either as ``"Mission"`` or ``"# Mission"``.
    Any heading level (``#``..``######``) at the start of a line counts.
    """
    if not sections:
        return [1.0 for _ in completions]

    cleaned: list[str] = []
    for s in sections:
        name = s.strip().lstrip("#").strip()
        if name:
            cleaned.append(name.lower())

    if not cleaned:
        return [1.0 for _ in completions]

    header_re = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)

    def score(c: str) -> float:
        found = {m.group(1).strip().lower() for m in header_re.finditer(c)}
        hits = sum(1 for required in cleaned if required in found)
        return hits / len(cleaned)

    return _safe_for_each(completions, score)


# ---------------------------------------------------------------------------
# Lexical
# ---------------------------------------------------------------------------


def lexical(
    completions: Sequence[str],
    *,
    required: list[str],
    forbidden: list[str],
    case_sensitive: bool = False,
) -> list[float]:
    """Score = required_hit_rate * forbidden_miss_rate.

    - ``required_hit_rate`` = fraction of required phrases present (1.0 if empty).
    - ``forbidden_miss_rate`` = fraction of forbidden phrases absent (1.0 if empty).
    """
    req = list(required or [])
    forb = list(forbidden or [])

    def normalise(s: str) -> str:
        return s if case_sensitive else s.lower()

    req_n = [normalise(r) for r in req if r]
    forb_n = [normalise(f) for f in forb if f]

    def score(c: str) -> float:
        hay = normalise(c)
        hit = sum(1 for r in req_n if r in hay) / len(req_n) if req_n else 1.0
        miss = sum(1 for f in forb_n if f not in hay) / len(forb_n) if forb_n else 1.0
        return hit * miss

    return _safe_for_each(completions, score)


# ---------------------------------------------------------------------------
# Length (triangular)
# ---------------------------------------------------------------------------


def length(
    completions: Sequence[str],
    *,
    min_tokens: int,
    max_tokens: int,
    sweet_spot: int,
) -> list[float]:
    """Triangular reward peaking at ``sweet_spot``, zero outside [min,max]."""
    if min_tokens < 0 or max_tokens < min_tokens or not (min_tokens <= sweet_spot <= max_tokens):
        # Degenerate config -> always zero; do not raise.
        logger.warning(
            "length: degenerate config min=%s max=%s sweet=%s",
            min_tokens,
            max_tokens,
            sweet_spot,
        )
        return [0.0 for _ in completions]

    def score(c: str) -> float:
        n = len(tokenize(c))
        if n < min_tokens or n > max_tokens:
            return 0.0
        if n == sweet_spot:
            return 1.0
        if n < sweet_spot:
            denom = sweet_spot - min_tokens
            return 0.0 if denom == 0 else (n - min_tokens) / denom
        denom = max_tokens - sweet_spot
        return 0.0 if denom == 0 else (max_tokens - n) / denom

    return _safe_for_each(completions, score)


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


def citation(
    completions: Sequence[str],
    *,
    require_ids: bool,
    min_citations: int,
    id_pattern: str = r"\[KNW-\d+\]",
) -> list[float]:
    """Reward presence of citation markers.

    - If ``require_ids`` is True and ``min_citations >= 1``, full credit requires
      at least ``min_citations`` matches of ``id_pattern``.
    - If ``require_ids`` is False, full credit is awarded when any bracketed
      reference (``[...]``) is present at the required count.
    - Partial credit is linear in the number of matches up to ``min_citations``.
    """
    target = max(0, int(min_citations))
    try:
        ids_re = re.compile(id_pattern)
    except re.error:
        logger.exception("citation: invalid id_pattern %r", id_pattern)
        return [0.0 for _ in completions]
    generic_re = re.compile(r"\[[^\]\n]{1,80}\]")

    def score(c: str) -> float:
        if target == 0:
            return 1.0
        matches = ids_re.findall(c) if require_ids else generic_re.findall(c)
        hit = min(len(matches), target)
        return hit / target

    return _safe_for_each(completions, score)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def safety(
    completions: Sequence[str],
    *,
    anti_pattern_terms: list[list[str]],
) -> list[float]:
    """Reward 1.0 if **no** anti-pattern phrase set fully matches; else 0.0.

    Each inner list represents a phrase set: ALL of its (case-insensitive)
    phrases must be present for the anti-pattern to be considered triggered.
    """
    sets: list[list[str]] = [
        [term.lower() for term in phrase_set if term]
        for phrase_set in (anti_pattern_terms or [])
        if phrase_set
    ]
    sets = [s for s in sets if s]

    def score(c: str) -> float:
        if not sets:
            return 1.0
        hay = c.lower()
        for phrase_set in sets:
            if all(term in hay for term in phrase_set):
                return 0.0
        return 1.0

    return _safe_for_each(completions, score)


# ---------------------------------------------------------------------------
# Principle judge
# ---------------------------------------------------------------------------


def _heuristic_overlap(completion: str, criterion: str) -> float:
    """Normalized token-overlap heuristic used as the default judge."""
    crit_tokens = set(tokenize(criterion))
    if not crit_tokens:
        return 0.0
    comp_tokens = set(tokenize(completion))
    if not comp_tokens:
        return 0.0
    inter = crit_tokens & comp_tokens
    return len(inter) / len(crit_tokens)


def principle_judge(
    completions: Sequence[str],
    prompts: Sequence[str] | None = None,
    *,
    principle_statement: str,
    judge: Callable[[Sequence[str], Sequence[str], str], list[float]] | None = None,
) -> list[float]:
    """Score each completion against ``principle_statement``.

    If a ``judge`` callable is provided, it receives ``(prompts, completions,
    principle_statement)`` and is expected to return ``list[float]`` of the same
    length. Otherwise we fall back to a normalized token-overlap heuristic so
    rewards remain useful (and tests stay deterministic) without an API key.
    """
    prompts = list(prompts or [""] * len(completions))
    if len(prompts) != len(completions):
        prompts = list(prompts) + [""] * (len(completions) - len(prompts))
        prompts = prompts[: len(completions)]

    if judge is not None:
        try:
            scores = judge(prompts, list(completions), principle_statement)
        except Exception:
            logger.exception("principle_judge: external judge crashed; falling back to heuristic")
            scores = [_heuristic_overlap(c, principle_statement) for c in completions]
        if len(scores) != len(completions):
            # Pad / truncate defensively.
            scores = list(scores)[: len(completions)]
            scores += [0.0] * (len(completions) - len(scores))
        return [_clip(float(s)) for s in scores]

    return _safe_for_each(completions, lambda c: _heuristic_overlap(c, principle_statement))


# ---------------------------------------------------------------------------
# Procedure check
# ---------------------------------------------------------------------------


def procedure_check(
    completions: Sequence[str],
    *,
    ordered_steps: list[str],
    ordered: bool = True,
) -> list[float]:
    """Score how well a completion follows a procedure's step keywords.

    Base score = fraction of step keywords present.
    If ``ordered``, scale down by the longest-increasing-subsequence ratio of
    matched step positions to penalize out-of-order matches.
    """
    steps = [s.strip().lower() for s in (ordered_steps or []) if s and s.strip()]
    if not steps:
        return [1.0 for _ in completions]

    def lis_length(seq: list[int]) -> int:
        # O(n log n) longest strictly increasing subsequence.
        if not seq:
            return 0
        from bisect import bisect_left

        tails: list[int] = []
        for x in seq:
            i = bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x
        return len(tails)

    def score(c: str) -> float:
        hay = c.lower()
        positions: list[int] = []
        hits = 0
        for step in steps:
            idx = hay.find(step)
            if idx >= 0:
                hits += 1
                positions.append(idx)
        if hits == 0:
            return 0.0
        base = hits / len(steps)
        if not ordered or len(positions) <= 1:
            return base
        # Order penalty: ratio of LIS length to number of hits.
        order_ratio = lis_length(positions) / len(positions)
        return base * order_ratio

    return _safe_for_each(completions, score)
