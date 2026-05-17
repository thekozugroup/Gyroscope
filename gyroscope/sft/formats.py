"""Serialise Trajectory objects to the common SFT row formats.

The module exposes both the per-format ``to_*`` / ``from_*`` helpers and
two registries — :data:`FORMAT_WRITERS` and :data:`FORMAT_READERS` — that
the pipeline and eval writers dispatch through. New formats register
themselves via :func:`register_format` so callers do not need to grow
if/elif chains when a new output shape is added.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gyroscope.core.models import Trajectory, TrajectoryMessage

# ShareGPT uses non-standard role names: human/gpt/system/tool.
_SHAREGPT_ROLE: dict[str, str] = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
    "tool": "tool",
}
_SHAREGPT_ROLE_INVERSE: dict[str, str] = {v: k for k, v in _SHAREGPT_ROLE.items()}


def _ensure_system_first(messages: list[TrajectoryMessage], system: str) -> list[TrajectoryMessage]:
    """Return messages with a leading system message guaranteed."""
    if messages and messages[0].role == "system":
        return list(messages)
    head = TrajectoryMessage(role="system", content=system)
    return [head, *messages]


def to_sharegpt(traj: Trajectory) -> dict[str, Any]:
    """Render a Trajectory as a ShareGPT row.

    Output shape:
        {"conversations": [{"from": "system|human|gpt|tool", "value": "..."}],
         "tags": {...},
         "id": "...",
         "scenario_id": "...",
         "quality_score": float | None}
    """
    messages = _ensure_system_first(traj.messages, traj.system)
    conversations: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {
            "from": _SHAREGPT_ROLE[m.role],
            "value": m.content,
        }
        if m.role == "tool" and m.name:
            entry["name"] = m.name
        conversations.append(entry)

    return {
        "id": traj.id,
        "scenario_id": traj.scenario_id,
        "conversations": conversations,
        "tags": dict(traj.tags),
        "quality_score": traj.quality_score,
    }


def to_chatml(traj: Trajectory) -> dict[str, Any]:
    """Render a Trajectory as a ChatML row.

    Output shape:
        {"messages": [{"role": "system|user|assistant|tool", "content": "..."}],
         "tags": {...},
         "id": "...",
         "scenario_id": "...",
         "quality_score": float | None}
    """
    messages = _ensure_system_first(traj.messages, traj.system)
    out_messages: list[dict[str, Any]] = []
    for m in messages:
        entry: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.role == "tool" and m.name:
            entry["name"] = m.name
        out_messages.append(entry)

    return {
        "id": traj.id,
        "scenario_id": traj.scenario_id,
        "messages": out_messages,
        "tags": dict(traj.tags),
        "quality_score": traj.quality_score,
    }


def to_alpaca(traj: Trajectory) -> dict[str, Any]:
    """Render a Trajectory as an Alpaca row.

    Multi-turn is collapsed:
      - `instruction`: the last user turn (the question the model must answer).
      - `input`:       prior conversational context (including system),
                       rendered as a labelled transcript. Empty if none.
      - `output`:      the final assistant turn.
    """
    messages = list(traj.messages)

    # Find last user message — that becomes the instruction.
    last_user_idx: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == "user":
            last_user_idx = idx
            break
    if last_user_idx is None:
        raise ValueError(f"Trajectory {traj.id} has no user turn; cannot convert to Alpaca.")

    # Find last assistant message — that becomes the output.
    last_assistant_idx: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == "assistant":
            last_assistant_idx = idx
            break
    if last_assistant_idx is None:
        raise ValueError(
            f"Trajectory {traj.id} has no assistant turn; cannot convert to Alpaca."
        )

    instruction = messages[last_user_idx].content
    output = messages[last_assistant_idx].content

    # Prior context = everything before the last user turn, plus any
    # assistant/tool turns that sit strictly between last_user and last_assistant
    # but exclude the final assistant turn itself.
    context_msgs: list[TrajectoryMessage] = []
    for idx, m in enumerate(messages):
        if idx in (last_user_idx, last_assistant_idx):
            continue
        if idx > last_assistant_idx:
            continue
        context_msgs.append(m)

    parts: list[str] = [f"SYSTEM: {traj.system}"] if traj.system else []
    for m in context_msgs:
        if m.role == "system":
            # Use traj.system as the canonical system; skip duplicate.
            if not traj.system:
                parts.append(f"SYSTEM: {m.content}")
        elif m.role == "user":
            parts.append(f"USER: {m.content}")
        elif m.role == "assistant":
            parts.append(f"ASSISTANT: {m.content}")
        elif m.role == "tool":
            label = f"TOOL[{m.name}]" if m.name else "TOOL"
            parts.append(f"{label}: {m.content}")

    return {
        "id": traj.id,
        "scenario_id": traj.scenario_id,
        "instruction": instruction,
        "input": "\n\n".join(parts),
        "output": output,
        "tags": dict(traj.tags),
        "quality_score": traj.quality_score,
    }


# ---------------------------------------------------------------------------
# Reverse parsers (used by the CLI report command to reload a written dataset).
# ---------------------------------------------------------------------------


def from_sharegpt(row: dict[str, Any]) -> Trajectory:
    """Parse a ShareGPT row back into a Trajectory.

    The system message is hoisted into Trajectory.system; remaining messages
    are returned in order. Unknown shareGPT roles raise.
    """
    conversations = row.get("conversations") or []
    system = ""
    messages: list[TrajectoryMessage] = []
    for entry in conversations:
        sg_role = entry.get("from", "")
        if sg_role not in _SHAREGPT_ROLE_INVERSE:
            raise ValueError(f"Unknown sharegpt role: {sg_role!r}")
        role = _SHAREGPT_ROLE_INVERSE[sg_role]
        content = entry.get("value", "")
        if role == "system" and not system:
            system = content
            continue
        msg = TrajectoryMessage(role=role, content=content, name=entry.get("name"))
        messages.append(msg)

    return Trajectory(
        id=row.get("id", "TRJ-unknown"),
        scenario_id=row.get("scenario_id", "SCN-unknown"),
        system=system,
        messages=messages,
        tags=dict(row.get("tags") or {}),
        quality_score=row.get("quality_score"),
    )


def from_chatml(row: dict[str, Any]) -> Trajectory:
    """Parse a ChatML row back into a Trajectory."""
    raw_messages = row.get("messages") or []
    system = ""
    messages: list[TrajectoryMessage] = []
    for entry in raw_messages:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "system" and not system:
            system = content
            continue
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unknown chatml role: {role!r}")
        messages.append(TrajectoryMessage(role=role, content=content, name=entry.get("name")))

    return Trajectory(
        id=row.get("id", "TRJ-unknown"),
        scenario_id=row.get("scenario_id", "SCN-unknown"),
        system=system,
        messages=messages,
        tags=dict(row.get("tags") or {}),
        quality_score=row.get("quality_score"),
    )


# ---------------------------------------------------------------------------
# Format registry. Dispatch happens through these dicts so adding a new
# format means a single :func:`register_format` call rather than editing
# every if/elif chain across the codebase.
# ---------------------------------------------------------------------------


FormatWriter = Callable[[Trajectory], dict[str, Any]]
FormatReader = Callable[[dict[str, Any]], Trajectory]


FORMAT_WRITERS: dict[str, FormatWriter] = {
    "sharegpt": to_sharegpt,
    "chatml": to_chatml,
    "alpaca": to_alpaca,
}

FORMAT_READERS: dict[str, FormatReader] = {
    "sharegpt": from_sharegpt,
    "chatml": from_chatml,
    # alpaca is lossy (multi-turn is collapsed); no reader.
}


def register_format(
    name: str,
    writer: FormatWriter,
    reader: FormatReader | None = None,
) -> None:
    """Register a new SFT output format.

    ``name`` is the string passed to :func:`render` and to
    ``SFTConfig.output_format``-style call sites. ``writer`` turns a
    Trajectory into a row dict. ``reader`` is optional — omit it for lossy
    formats where a Trajectory cannot be reconstructed (Alpaca is the
    canonical example). Registering an existing ``name`` overwrites the
    previous binding so callers can monkeypatch a format in tests.
    """
    FORMAT_WRITERS[name] = writer
    if reader is not None:
        FORMAT_READERS[name] = reader


def render(traj: Trajectory, output_format: str) -> dict[str, Any]:
    """Render a trajectory in the requested format.

    Dispatches through :data:`FORMAT_WRITERS` so any format registered via
    :func:`register_format` is automatically supported here too.
    """
    try:
        writer = FORMAT_WRITERS[output_format]
    except KeyError as exc:
        raise ValueError(
            f"Unknown output_format: {output_format!r}. "
            f"Registered formats: {sorted(FORMAT_WRITERS)}"
        ) from exc
    return writer(traj)
