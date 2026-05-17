"""Render a QualityReport as markdown or self-contained HTML."""

from __future__ import annotations

import html
import json
from pathlib import Path

from gyroscope.quality.metrics import QualityReport


def render_report_markdown(report: QualityReport) -> str:
    lines = ["# Gyroscope Quality Report\n"]
    lines.append(f"**Overall:** {report.overall():.1f} / 100\n")
    lines.append("| Axis | Score | Notes |")
    lines.append("|------|-------|-------|")
    for name, axis in report.axes.items():
        notes_joined = "<br>".join(axis.notes) if axis.notes else ""
        lines.append(f"| {name} | {axis.score:.1f} | {notes_joined} |")
    lines.append("")
    for name, axis in report.axes.items():
        lines.append(f"## {name} — {axis.score:.1f}\n")
        for n in axis.notes:
            lines.append(f"- {n}")
        if axis.details:
            lines.append("\n```json")
            lines.append(json.dumps(axis.details, indent=2, default=str))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_report_html(report: QualityReport) -> str:
    overall = report.overall()
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 880px; margin: 2rem auto; padding: 0 1rem; color: #1f2328; }
    h1 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; }
    .overall { font-size: 2rem; font-weight: 600; }
    .axes { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem; }
    .score { font-size: 1.4rem; font-weight: 600; }
    .pass { color: #1a7f37; } .warn { color: #bf8700; } .fail { color: #cf222e; }
    pre { background: #f6f8fa; padding: 0.75rem; border-radius: 6px; overflow-x: auto; }
    ul { margin: 0.4rem 0 0.4rem 1.2rem; padding: 0; }
    """
    cards = []
    for name, axis in report.axes.items():
        cls = "pass" if axis.score >= 95 else "warn" if axis.score >= 80 else "fail"
        notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in axis.notes)
        details_html = (
            f"<pre>{html.escape(json.dumps(axis.details, indent=2, default=str))}</pre>"
            if axis.details
            else ""
        )
        cards.append(
            f'<div class="card"><h3>{html.escape(name)}</h3>'
            f'<div class="score {cls}">{axis.score:.1f}</div>'
            f"<ul>{notes_html}</ul>{details_html}</div>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Gyroscope Quality Report</title>"
        f"<style>{css}</style></head><body>"
        "<h1>Gyroscope Quality Report</h1>"
        f"<div class='overall'>{overall:.1f} / 100</div>"
        f"<div class='axes'>{''.join(cards)}</div>"
        "</body></html>"
    )


def write_report_artefacts(report: QualityReport, out_dir: Path | str) -> tuple[Path, Path, Path]:
    """Write ``report.{md,html,json}`` into ``out_dir`` and return the paths.

    Single shared writer used by both :class:`AutonomousRunner` and the
    ``gyroscope report`` CLI so the on-disk filename convention and
    serialisation format cannot drift between them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "report.md"
    html_path = out_dir / "report.html"
    json_path = out_dir / "report.json"
    md.write_text(render_report_markdown(report), encoding="utf-8")
    html_path.write_text(render_report_html(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return md, html_path, json_path
