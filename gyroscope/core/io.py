"""IO helpers: JSONL, YAML, markdown rendering of the golden document."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows to a JSONL file. Returns the row count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def write_yaml(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def read_yaml(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Markdown rendering of GoldenDocument
# ---------------------------------------------------------------------------


def golden_to_markdown(golden: Any) -> str:
    """Render a GoldenDocument to canonical markdown."""
    # Imported here to avoid circular import with models.
    from gyroscope.core.models import GoldenDocument

    if not isinstance(golden, GoldenDocument):
        raise TypeError(f"Expected GoldenDocument, got {type(golden).__name__}")

    lines: list[str] = []
    g = golden

    lines.append("# Identity\n")
    lines.append(f"**Role:** {g.identity.role}\n")
    lines.append(f"{g.identity.description}\n")

    lines.append("# Mission\n")
    lines.append(f"{g.identity.mission}\n")

    if g.principles:
        lines.append("# Principles\n")
        for p in g.principles:
            lines.append(f"- **[{p.id}]** {p.statement}")
            if p.rationale:
                lines.append(f"  - _why:_ {p.rationale}")
        lines.append("")

    if g.procedures:
        lines.append("# Procedures\n")
        for proc in g.procedures:
            lines.append(f"## [{proc.id}] {proc.name}\n")
            lines.append(f"_Purpose:_ {proc.purpose}\n")
            if proc.preconditions:
                lines.append("**Preconditions:**")
                for pc in proc.preconditions:
                    lines.append(f"- {pc}")
                lines.append("")
            lines.append("**Steps:**")
            for s in sorted(proc.steps, key=lambda x: x.order):
                tool = f" _(tool: {s.tool})_" if s.tool else ""
                lines.append(f"{s.order}. {s.action}{tool}")
                if s.expected_output:
                    lines.append(f"   - expects: {s.expected_output}")
            if proc.postconditions:
                lines.append("\n**Postconditions:**")
                for pc in proc.postconditions:
                    lines.append(f"- {pc}")
            lines.append("")

    if g.knowledge:
        lines.append("# Knowledge\n")
        for k in g.knowledge:
            tags = f" `{' '.join(k.tags)}`" if k.tags else ""
            lines.append(f"- **[{k.id}]**{tags} {k.statement}")
        lines.append("")

    if g.vocabulary:
        lines.append("# Vocabulary\n")
        for v in g.vocabulary:
            aliases = f" _(aka: {', '.join(v.aliases)})_" if v.aliases else ""
            lines.append(f"- **{v.term}**{aliases}: {v.definition}")
        lines.append("")

    if g.anti_patterns:
        lines.append("# Anti-patterns\n")
        for a in g.anti_patterns:
            lines.append(f"## [{a.id}]\n")
            lines.append(f"**Don't:** {a.description}\n")
            lines.append(f"**Why:** {a.why_bad}\n")
            lines.append(f"**Instead:** {a.correction}\n")

    if g.source_documents:
        lines.append("# Sources\n")
        for src in g.source_documents:
            lines.append(f"- {src}")

    return "\n".join(lines).rstrip() + "\n"
