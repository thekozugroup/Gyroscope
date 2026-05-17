"""Exhaustive tests for the reward primitives in gyroscope.rewards.library."""

from __future__ import annotations

import pytest

from gyroscope.rewards import library as lib

# ---------------------------------------------------------------------------
# format_regex
# ---------------------------------------------------------------------------


class TestFormatRegex:
    def test_happy_path_match(self) -> None:
        scores = lib.format_regex(["abc-123", "no number here"], pattern=r"[a-z]+-\d+")
        assert scores == [1.0, 0.0]

    def test_empty_completions(self) -> None:
        assert lib.format_regex([], pattern=r".") == []

    def test_invalid_regex_returns_zeros(self) -> None:
        scores = lib.format_regex(["hi", "there"], pattern="(unclosed")
        assert scores == [0.0, 0.0]

    def test_multiline_dotall(self) -> None:
        scores = lib.format_regex(["line1\nline2"], pattern=r"^line1.*line2$")
        assert scores == [1.0]

    def test_non_string_row_is_zero(self) -> None:
        # type: ignore[list-item] for malformed input simulation
        scores = lib.format_regex([None, "ok"], pattern=r"ok")  # type: ignore[list-item]
        assert scores == [0.0, 1.0]


# ---------------------------------------------------------------------------
# format_json_schema
# ---------------------------------------------------------------------------


_SCHEMA: dict = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}


class TestFormatJsonSchema:
    def test_valid_json(self) -> None:
        ok = '{"name": "alice", "age": 30}'
        bad = '{"name": "bob"}'
        not_json = "definitely not json"
        scores = lib.format_json_schema([ok, bad, not_json], schema=_SCHEMA)
        assert scores == [1.0, 0.0, 0.0]

    def test_empty(self) -> None:
        assert lib.format_json_schema([], schema=_SCHEMA) == []


# ---------------------------------------------------------------------------
# format_sections
# ---------------------------------------------------------------------------


class TestFormatSections:
    def test_all_present(self) -> None:
        text = "# Intro\nsome\n## Body\nmore\n### Conclusion\nend"
        scores = lib.format_sections([text], sections=["Intro", "Body", "Conclusion"])
        assert scores == [1.0]

    def test_partial(self) -> None:
        text = "# Intro\nsome\n## Body\nmore"
        scores = lib.format_sections([text], sections=["Intro", "Body", "Conclusion"])
        assert scores == [pytest.approx(2 / 3)]

    def test_missing_all(self) -> None:
        scores = lib.format_sections(["plain text"], sections=["A", "B"])
        assert scores == [0.0]

    def test_accepts_hashed_section_arg(self) -> None:
        text = "# Foo\nbody"
        scores = lib.format_sections([text], sections=["# Foo"])
        assert scores == [1.0]

    def test_empty_sections_is_full_credit(self) -> None:
        assert lib.format_sections(["anything"], sections=[]) == [1.0]


# ---------------------------------------------------------------------------
# lexical
# ---------------------------------------------------------------------------


class TestLexical:
    def test_required_only(self) -> None:
        text = "use the proper procedure"
        s = lib.lexical(
            [text, "irrelevant"],
            required=["procedure", "proper"],
            forbidden=[],
        )
        assert s[0] == pytest.approx(1.0)
        assert s[1] == pytest.approx(0.0)

    def test_forbidden_only(self) -> None:
        s = lib.lexical(["clean", "contains badword"], required=[], forbidden=["badword"])
        assert s == [1.0, 0.0]

    def test_combined_multiplicative(self) -> None:
        s = lib.lexical(
            ["foo bar baz"],
            required=["foo", "bar"],
            forbidden=["baz"],
        )
        # required_hit_rate = 1, forbidden_miss_rate = 0 -> 0
        assert s == [0.0]

    def test_case_sensitive(self) -> None:
        s = lib.lexical(
            ["Hello World"],
            required=["hello"],
            forbidden=[],
            case_sensitive=True,
        )
        assert s == [0.0]

    def test_empty_completions(self) -> None:
        assert lib.lexical([], required=["x"], forbidden=[]) == []


# ---------------------------------------------------------------------------
# length (triangular)
# ---------------------------------------------------------------------------


class TestLength:
    def test_peaks_at_sweet_spot(self) -> None:
        text = " ".join(["w"] * 50)
        s = lib.length([text], min_tokens=10, max_tokens=100, sweet_spot=50)
        assert s == [1.0]

    def test_below_min_zero(self) -> None:
        s = lib.length(["one two three"], min_tokens=10, max_tokens=100, sweet_spot=50)
        assert s == [0.0]

    def test_above_max_zero(self) -> None:
        text = " ".join(["w"] * 200)
        s = lib.length([text], min_tokens=10, max_tokens=100, sweet_spot=50)
        assert s == [0.0]

    def test_linear_ramps(self) -> None:
        before = " ".join(["w"] * 30)  # halfway between min=10 and sweet=50
        after = " ".join(["w"] * 75)  # halfway between sweet=50 and max=100
        before_s = lib.length([before], min_tokens=10, max_tokens=100, sweet_spot=50)[0]
        after_s = lib.length([after], min_tokens=10, max_tokens=100, sweet_spot=50)[0]
        assert before_s == pytest.approx(0.5)
        assert after_s == pytest.approx(0.5)

    def test_degenerate_config_returns_zeros(self) -> None:
        s = lib.length(["abc"], min_tokens=100, max_tokens=10, sweet_spot=50)
        assert s == [0.0]

    def test_empty_completions(self) -> None:
        assert lib.length([], min_tokens=0, max_tokens=10, sweet_spot=5) == []


# ---------------------------------------------------------------------------
# citation
# ---------------------------------------------------------------------------


class TestCitation:
    def test_id_pattern_full_credit(self) -> None:
        s = lib.citation(
            ["see [KNW-0001] and [KNW-0042]"],
            require_ids=True,
            min_citations=2,
        )
        assert s == [1.0]

    def test_id_pattern_partial(self) -> None:
        s = lib.citation(
            ["see [KNW-0001]"],
            require_ids=True,
            min_citations=2,
        )
        assert s == [0.5]

    def test_no_citation_zero(self) -> None:
        s = lib.citation(["no refs"], require_ids=True, min_citations=1)
        assert s == [0.0]

    def test_generic_bracket_when_not_requiring_ids(self) -> None:
        s = lib.citation(["see [Smith 2020]"], require_ids=False, min_citations=1)
        assert s == [1.0]

    def test_min_zero_always_full_credit(self) -> None:
        s = lib.citation(["nothing"], require_ids=True, min_citations=0)
        assert s == [1.0]

    def test_custom_pattern(self) -> None:
        s = lib.citation(
            ["see [REF-7]"],
            require_ids=True,
            min_citations=1,
            id_pattern=r"\[REF-\d+\]",
        )
        assert s == [1.0]


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------


class TestSafety:
    def test_no_match_is_one(self) -> None:
        s = lib.safety(["a clean response"], anti_pattern_terms=[["weapon", "make"]])
        assert s == [1.0]

    def test_full_set_match_is_zero(self) -> None:
        s = lib.safety(
            ["how to make a weapon at home"],
            anti_pattern_terms=[["weapon", "make"]],
        )
        assert s == [0.0]

    def test_partial_match_does_not_trigger(self) -> None:
        s = lib.safety(
            ["how to make a sandwich"],
            anti_pattern_terms=[["weapon", "make"]],
        )
        assert s == [1.0]

    def test_overlapping_sets_one_match_enough(self) -> None:
        s = lib.safety(
            ["foo bar baz"],
            anti_pattern_terms=[["foo", "bar"], ["xyz", "abc"]],
        )
        assert s == [0.0]

    def test_empty_anti_patterns(self) -> None:
        s = lib.safety(["anything"], anti_pattern_terms=[])
        assert s == [1.0]


# ---------------------------------------------------------------------------
# principle_judge
# ---------------------------------------------------------------------------


class TestPrincipleJudge:
    def test_heuristic_overlap(self) -> None:
        s = lib.principle_judge(
            ["alpha beta gamma", "delta epsilon"],
            ["q", "q"],
            principle_statement="alpha beta",
        )
        assert s[0] == pytest.approx(1.0)
        assert s[1] == pytest.approx(0.0)

    def test_injected_judge_used(self) -> None:
        calls: list[tuple[list[str], list[str], str]] = []

        def fake(prompts, completions, criterion):
            calls.append((list(prompts), list(completions), criterion))
            return [0.42 for _ in completions]

        s = lib.principle_judge(
            ["c1", "c2"],
            ["p1", "p2"],
            principle_statement="be kind",
            judge=fake,
        )
        assert s == [0.42, 0.42]
        assert calls == [(["p1", "p2"], ["c1", "c2"], "be kind")]

    def test_judge_crashes_falls_back(self) -> None:
        def bad(prompts, completions, criterion):
            raise RuntimeError("boom")

        s = lib.principle_judge(
            ["alpha"],
            ["p"],
            principle_statement="alpha",
            judge=bad,
        )
        assert s == [pytest.approx(1.0)]

    def test_prompts_padded_when_short(self) -> None:
        s = lib.principle_judge(
            ["c1", "c2"],
            ["only one"],
            principle_statement="c1",
        )
        assert len(s) == 2


# ---------------------------------------------------------------------------
# procedure_check
# ---------------------------------------------------------------------------


class TestProcedureCheck:
    def test_all_in_order(self) -> None:
        s = lib.procedure_check(
            ["First gather requirements then design then ship"],
            ordered_steps=["gather", "design", "ship"],
            ordered=True,
        )
        assert s == [pytest.approx(1.0)]

    def test_partial(self) -> None:
        s = lib.procedure_check(
            ["gather then ship"],
            ordered_steps=["gather", "design", "ship"],
            ordered=True,
        )
        assert s[0] == pytest.approx(2 / 3)

    def test_unordered_full_credit_when_present(self) -> None:
        s = lib.procedure_check(
            ["ship design gather"],
            ordered_steps=["gather", "design", "ship"],
            ordered=False,
        )
        assert s == [pytest.approx(1.0)]

    def test_ordered_penalises_out_of_order(self) -> None:
        ordered = lib.procedure_check(
            ["ship design gather"],
            ordered_steps=["gather", "design", "ship"],
            ordered=True,
        )[0]
        # All three present (base 1.0), but LIS of positions is 1 -> 1/3.
        assert ordered == pytest.approx(1 / 3)

    def test_no_keywords_full_credit(self) -> None:
        s = lib.procedure_check(["anything"], ordered_steps=[])
        assert s == [1.0]
