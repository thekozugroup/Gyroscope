"""Tests for report rendering."""

from __future__ import annotations

from gyroscope.quality.metrics import AxisScore, QualityReport
from gyroscope.quality.report import render_report_html, render_report_markdown


def _report() -> QualityReport:
    r = QualityReport()
    r.add(AxisScore("coverage", 95.0, ["good"], {"x": 1}))
    r.add(AxisScore("faithfulness", 88.0, ["ok"], {"y": 2}))
    return r


def test_markdown_render_contains_axes():
    md = render_report_markdown(_report())
    assert "# Gyroscope Quality Report" in md
    assert "## coverage" in md
    assert "## faithfulness" in md
    assert "Overall" in md


def test_html_render_is_self_contained():
    html_text = render_report_html(_report())
    assert html_text.startswith("<!doctype html>")
    assert "Quality Report" in html_text
    assert "coverage" in html_text
    assert "faithfulness" in html_text


def test_overall_and_min_axis():
    r = _report()
    assert 90 < r.overall() < 95
    m = r.min_axis()
    assert m is not None and m.name == "faithfulness"


def test_all_pass_threshold():
    r = _report()
    assert r.all_pass(80.0) is True
    assert r.all_pass(95.0) is False
