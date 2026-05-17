"""LLM-driven extractors that turn deduped chunks into the typed pieces of
a :class:`GoldenDocument`.

Each extractor:

* Batches chunks so the prompt fits ~80% of an Opus 200K window. With
  ``CurationConfig.chunk_target_tokens = 1200`` that means we can pack
  ~120 chunks per call; we conservatively use ``DEFAULT_BATCH_SIZE = 24``
  to leave room for the system prompt, response, and chunk metadata.
* Uses ``LLMClient.complete_json_array`` with a strict instruction that
  the model must emit a JSON array matching the documented schema and
  must only assert things grounded in the provided chunks.
* Caches its system prompt (``cache_system=True``) since it is identical
  across batches.
* Assigns deterministic ids based on input order — never UUIDs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from gyroscope.core.llm import LLMClient
from gyroscope.core.models import (
    AntiPattern,
    Chunk,
    Identity,
    KnowledgeItem,
    Principle,
    Procedure,
    ProcedureStep,
    VocabularyTerm,
)

logger = logging.getLogger(__name__)


# A conservative batch size. The Opus 200K window minus ~20% headroom for
# system + response gives us ~160K tokens to spend on chunks. With chunks
# averaging ~1200 tokens we could fit ~130 per batch, but we keep the
# batch small enough that a single call also keeps the JSON response
# manageable and easy to repair if the model truncates.
DEFAULT_BATCH_SIZE = 24

# Top-N chunks used to seed Identity extraction. The first chunks of a
# document typically carry titles, scope, and intro material.
IDENTITY_REPRESENTATIVE_N = 12


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


_FAITHFULNESS_RULES = """\
You are extracting structured knowledge from a Body-of-Knowledge corpus.

Hard rules:
- Use ONLY information present in the provided source chunks. Do not invent
  facts, do not fill in gaps from general knowledge.
- Every emitted item MUST cite at least one source chunk id in the
  appropriate field (`source_chunk_ids` or `citations`).
- If a given chunk contains nothing relevant, simply emit nothing for it.
- Output MUST be a single JSON array. No prose, no markdown, no code fence.
- Each item MUST validate against the schema described below.
- Prefer concise, atomic statements over long paragraphs.
"""


_PRINCIPLE_SCHEMA = """\
Each item is an object with:
- `statement` (string, REQUIRED): one atomic, testable rule the role must
  follow. One sentence, present tense.
- `rationale` (string, optional): a brief justification.
- `source_chunk_ids` (array of strings, REQUIRED): chunk ids supporting it.
- `weight` (number, optional, default 1.0).
"""


_PROCEDURE_SCHEMA = """\
Each item is an object with:
- `name` (string, REQUIRED): short imperative title.
- `purpose` (string, REQUIRED): one-sentence purpose.
- `steps` (array, REQUIRED, ordered): each step is
  `{ "order": int, "action": str, "expected_output": str|null, "tool": str|null }`.
  `order` starts at 1 and must be contiguous.
- `preconditions` (array of strings, optional).
- `postconditions` (array of strings, optional).
- `source_chunk_ids` (array of strings, REQUIRED).
"""


_KNOWLEDGE_SCHEMA = """\
Each item is an object with:
- `statement` (string, REQUIRED): one declarative fact.
- `citations` (array of strings, REQUIRED): chunk ids backing the fact.
- `tags` (array of strings, optional).
"""


_VOCAB_SCHEMA = """\
Each item is an object with:
- `term` (string, REQUIRED).
- `definition` (string, REQUIRED).
- `aliases` (array of strings, optional).
"""


_ANTI_SCHEMA = """\
Each item is an object with:
- `description` (string, REQUIRED): what NOT to do.
- `why_bad` (string, REQUIRED): why it is harmful or wrong.
- `correction` (string, REQUIRED): what to do instead.
- `source_chunk_ids` (array of strings, REQUIRED).
"""


def _principle_system_prompt() -> str:
    return f"{_FAITHFULNESS_RULES}\nReturn an array of PRINCIPLE objects.\n{_PRINCIPLE_SCHEMA}"


def _procedure_system_prompt() -> str:
    return f"{_FAITHFULNESS_RULES}\nReturn an array of PROCEDURE objects.\n{_PROCEDURE_SCHEMA}"


def _knowledge_system_prompt() -> str:
    return f"{_FAITHFULNESS_RULES}\nReturn an array of KNOWLEDGE objects.\n{_KNOWLEDGE_SCHEMA}"


def _vocabulary_system_prompt() -> str:
    return f"{_FAITHFULNESS_RULES}\nReturn an array of VOCABULARY objects.\n{_VOCAB_SCHEMA}"


def _anti_pattern_system_prompt() -> str:
    return f"{_FAITHFULNESS_RULES}\nReturn an array of ANTI-PATTERN objects.\n{_ANTI_SCHEMA}"


def _identity_system_prompt() -> str:
    return (
        f"{_FAITHFULNESS_RULES}\n"
        "Return a SINGLE JSON object describing the role this corpus equips.\n"
        "Schema:\n"
        "- `role` (string, REQUIRED): one-line role title.\n"
        "- `description` (string, REQUIRED): 2-4 sentences describing the role.\n"
        "- `mission` (string, REQUIRED): one-sentence mission statement.\n"
    )


def _render_chunks(chunks: Iterable[Chunk]) -> str:
    """Render a batch of chunks into the user prompt. Each chunk is wrapped
    with a banner naming its id and source so the model can cite them."""
    out: list[str] = []
    for chunk in chunks:
        out.append(
            f'<<CHUNK id="{chunk.id}" source="{chunk.document_source}">>\n'
            f"{chunk.text}\n"
            f"<<END CHUNK {chunk.id}>>"
        )
    return "\n\n".join(out)


def _batched(items: list[Chunk], size: int) -> list[list[Chunk]]:
    if size <= 0:
        raise ValueError("Batch size must be positive.")
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# ID assignment
# ---------------------------------------------------------------------------


def _assign_ids(prefix: str, items: list[dict[str, Any]]) -> None:
    """Mutate items in-place, adding a deterministic ``id`` field."""
    for i, item in enumerate(items, start=1):
        item["id"] = f"{prefix}-{i:04d}"


# ---------------------------------------------------------------------------
# Sanitisation: filter cited chunk ids to known set so synthesizer ranking
# stays honest, and drop items that cite nothing in the known set.
# ---------------------------------------------------------------------------


def _filter_citations(
    items: list[dict[str, Any]],
    *,
    known_ids: set[str],
    citation_key: str,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in items:
        raw = item.get(citation_key) or []
        if not isinstance(raw, list):
            continue
        cleaned = [c for c in raw if isinstance(c, str) and c in known_ids]
        if not cleaned:
            # No grounded citation — skip the item.
            continue
        item[citation_key] = cleaned
        kept.append(item)
    return kept


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _default_identity() -> Identity:
    return Identity(
        role="Unspecified Role",
        description="No source material available.",
        mission="No mission defined.",
    )


async def extract_identity(chunks: list[Chunk], client: LLMClient) -> Identity:
    """Build a single Identity from the top-N representative chunks.

    If the model emits a malformed payload that ``complete_json`` cannot
    parse, fall back to the same default identity used for the empty-chunks
    branch — a single bad output must not crash the curation phase.
    """
    if not chunks:
        return _default_identity()

    representatives = chunks[:IDENTITY_REPRESENTATIVE_N]
    system = _identity_system_prompt()
    user = (
        "From the following chunks, derive the role's Identity. "
        "Return a single JSON object as described.\n\n"
        f"{_render_chunks(representatives)}"
    )

    fallback_payload: dict[str, str] = {
        "role": "Unspecified Role",
        "description": "No source material available.",
        "mission": "No mission defined.",
    }
    try:
        payload = await client.complete_json(
            system=system,
            user=user,
            model=client.model_for("curator"),
            temperature=client.temperature_for("curator"),
            cache_system=True,
            strict=False,
            default=fallback_payload,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Identity extraction failed to parse model output: %s", exc)
        return _default_identity()

    if not isinstance(payload, dict):
        logger.warning(
            "Identity extraction expected an object, got %s; falling back to default.",
            type(payload).__name__,
        )
        return _default_identity()

    try:
        return Identity(
            role=str(payload.get("role", "Unspecified Role")).strip() or "Unspecified Role",
            description=str(payload.get("description", "")).strip() or "No description provided.",
            mission=str(payload.get("mission", "")).strip() or "No mission defined.",
        )
    except ValidationError as exc:
        logger.warning("Identity payload failed validation: %s; falling back to default.", exc)
        return _default_identity()


# ---------------------------------------------------------------------------
# Generic batched extractor used by principle / knowledge / vocabulary /
# procedure / anti-pattern flows.
# ---------------------------------------------------------------------------


async def _run_batched_array(
    *,
    chunks: list[Chunk],
    client: LLMClient,
    system_prompt: str,
    batch_size: int,
    instruction: str,
) -> list[dict[str, Any]]:
    """Issue one LLM call per batch concurrently and flatten the arrays.

    Batches are dispatched in parallel via ``asyncio.gather`` so a single
    extractor can saturate the ``LLMClient`` semaphore on its own; the
    semaphore already caps in-flight requests across all extractors, so no
    additional limiter is needed here. ``gather`` preserves input order,
    which we rely on when flattening to keep deterministic id assignment.
    """
    if not chunks:
        return []

    model = client.model_for("curator")
    temperature = client.temperature_for("curator")
    batches = _batched(chunks, batch_size)

    async def _run_one(batch: list[Chunk]) -> list[dict[str, Any]]:
        user = f"{instruction}\n\n{_render_chunks(batch)}"
        raw = await client.complete_json_array(
            system=system_prompt,
            user=user,
            model=model,
            temperature=temperature,
            cache_system=True,
        )
        if not isinstance(raw, list):
            logger.warning("Batched extractor returned non-list payload; skipping batch.")
            return []
        return [entry for entry in raw if isinstance(entry, dict)]

    batch_results = await asyncio.gather(*(_run_one(batch) for batch in batches))

    results: list[dict[str, Any]] = []
    for entries in batch_results:
        results.extend(entries)
    return results


# ---------------------------------------------------------------------------
# Principles
# ---------------------------------------------------------------------------


async def extract_principles(
    chunks: list[Chunk],
    client: LLMClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Principle]:
    known = {c.id for c in chunks}
    raw = await _run_batched_array(
        chunks=chunks,
        client=client,
        system_prompt=_principle_system_prompt(),
        batch_size=batch_size,
        instruction=(
            "Extract every atomic, testable principle the role MUST follow that "
            "is stated or strongly implied by the chunks below. One principle per "
            "object. Aim for at most 10 per chunk."
        ),
    )
    raw = _filter_citations(raw, known_ids=known, citation_key="source_chunk_ids")
    # Local within-batch text dedup (exact / near-exact statement).
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        statement = str(item.get("statement", "")).strip()
        if not statement:
            continue
        key = _normalise_statement(statement)
        if key in seen:
            continue
        seen.add(key)
        item["statement"] = statement
        deduped.append(item)

    _assign_ids("PRN", deduped)

    out: list[Principle] = []
    for item in deduped:
        try:
            out.append(
                Principle(
                    id=item["id"],
                    statement=item["statement"],
                    rationale=_optional_str(item.get("rationale")),
                    source_chunk_ids=list(item.get("source_chunk_ids", [])),
                    weight=float(item.get("weight", 1.0) or 1.0),
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed principle %s: %s", item.get("id"), exc)
    return out


# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------


async def extract_procedures(
    chunks: list[Chunk],
    client: LLMClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[Procedure], int]:
    """Extract procedures from chunks.

    Returns a ``(procedures, dropped_zero_step_count)`` tuple. The second
    element is the number of procedures emitted by the model that we had to
    drop because they parsed with zero valid steps — surfaced for telemetry
    rather than via a module-level global (which would race when two
    extractions ran concurrently).
    """
    known = {c.id for c in chunks}
    raw = await _run_batched_array(
        chunks=chunks,
        client=client,
        system_prompt=_procedure_system_prompt(),
        batch_size=batch_size,
        instruction=(
            "Extract every procedure or step-by-step playbook described in the "
            "chunks below. Procedures must have ordered, contiguous steps starting "
            "at 1."
        ),
    )
    raw = _filter_citations(raw, known_ids=known, citation_key="source_chunk_ids")
    _assign_ids("PRC", raw)

    out: list[Procedure] = []
    dropped = 0
    for item in raw:
        try:
            steps_payload = item.get("steps") or []
            steps: list[ProcedureStep] = []
            for j, step_payload in enumerate(steps_payload, start=1):
                if not isinstance(step_payload, dict):
                    continue
                steps.append(
                    ProcedureStep(
                        order=int(step_payload.get("order", j)),
                        action=str(step_payload.get("action", "")).strip(),
                        expected_output=_optional_str(step_payload.get("expected_output")),
                        tool=_optional_str(step_payload.get("tool")),
                    )
                )
            if not steps:
                dropped += 1
                continue
            out.append(
                Procedure(
                    id=item["id"],
                    name=str(item.get("name", "")).strip() or item["id"],
                    purpose=str(item.get("purpose", "")).strip() or "Unspecified purpose.",
                    steps=steps,
                    preconditions=[str(x) for x in item.get("preconditions") or []],
                    postconditions=[str(x) for x in item.get("postconditions") or []],
                    source_chunk_ids=list(item.get("source_chunk_ids", [])),
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed procedure %s: %s", item.get("id"), exc)
    return out, dropped


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------


async def extract_knowledge(
    chunks: list[Chunk],
    client: LLMClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[KnowledgeItem]:
    known = {c.id for c in chunks}
    raw = await _run_batched_array(
        chunks=chunks,
        client=client,
        system_prompt=_knowledge_system_prompt(),
        batch_size=batch_size,
        instruction=(
            "Extract every standalone factual statement the role should know "
            "from the chunks below. One declarative fact per object."
        ),
    )
    raw = _filter_citations(raw, known_ids=known, citation_key="citations")

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        statement = str(item.get("statement", "")).strip()
        if not statement:
            continue
        key = _normalise_statement(statement)
        if key in seen:
            continue
        seen.add(key)
        item["statement"] = statement
        deduped.append(item)

    _assign_ids("KNW", deduped)

    out: list[KnowledgeItem] = []
    for item in deduped:
        try:
            out.append(
                KnowledgeItem(
                    id=item["id"],
                    statement=item["statement"],
                    citations=list(item.get("citations", [])),
                    tags=[str(t) for t in item.get("tags") or []],
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed knowledge item %s: %s", item.get("id"), exc)
    return out


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


async def extract_vocabulary(
    chunks: list[Chunk],
    client: LLMClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[VocabularyTerm]:
    raw = await _run_batched_array(
        chunks=chunks,
        client=client,
        system_prompt=_vocabulary_system_prompt(),
        batch_size=batch_size,
        instruction=(
            "Extract domain-specific terminology defined in or essential to "
            "understanding the chunks below. One term per object."
        ),
    )

    seen: set[str] = set()
    out: list[VocabularyTerm] = []
    for item in raw:
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if not term or not definition:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            out.append(
                VocabularyTerm(
                    term=term,
                    definition=definition,
                    aliases=[str(a) for a in item.get("aliases") or []],
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed vocabulary term %s: %s", term, exc)
    return out


# ---------------------------------------------------------------------------
# Anti-patterns
# ---------------------------------------------------------------------------


async def extract_anti_patterns(
    chunks: list[Chunk],
    client: LLMClient,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[AntiPattern]:
    known = {c.id for c in chunks}
    raw = await _run_batched_array(
        chunks=chunks,
        client=client,
        system_prompt=_anti_pattern_system_prompt(),
        batch_size=batch_size,
        instruction=(
            "Extract every anti-pattern, common mistake, or behaviour the role "
            "must avoid that is documented in the chunks below."
        ),
    )
    raw = _filter_citations(raw, known_ids=known, citation_key="source_chunk_ids")
    _assign_ids("ANT", raw)

    out: list[AntiPattern] = []
    for item in raw:
        try:
            out.append(
                AntiPattern(
                    id=item["id"],
                    description=str(item.get("description", "")).strip(),
                    why_bad=str(item.get("why_bad", "")).strip(),
                    correction=str(item.get("correction", "")).strip(),
                    source_chunk_ids=list(item.get("source_chunk_ids", [])),
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed anti-pattern %s: %s", item.get("id"), exc)
    return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")


def _normalise_statement(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s.lower()).strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None
