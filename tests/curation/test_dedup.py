"""Tests for the MinHash-based Deduplicator."""

from __future__ import annotations

import pytest

from gyroscope.core.config import CurationConfig
from gyroscope.core.models import Chunk
from gyroscope.curation.dedup import Deduplicator


def _chunk(cid: str, text: str, order: int = 0) -> Chunk:
    return Chunk(id=cid, document_source="src://x", text=text, order=order)


def test_dedup_collapses_exact_duplicates_first_wins():
    text = "The quantity surveyor must verify quantities against drawings and specifications."
    chunks = [
        _chunk("a-0001", text, order=0),
        _chunk("a-0002", text, order=1),
        _chunk("a-0003", text, order=2),
    ]
    deduper = Deduplicator(CurationConfig(dedup_threshold=0.85))
    kept = deduper.dedupe(chunks)
    assert len(kept) == 1
    assert kept[0].id == "a-0001"


def test_dedup_keeps_distinct_chunks():
    chunks = [
        _chunk("a-0001", "Concrete strength is measured in megapascals after curing for 28 days."),
        _chunk("a-0002", "Steel reinforcement bars provide tensile strength in concrete elements."),
        _chunk("a-0003", "The site engineer schedules concrete pours during dry weather windows."),
    ]
    deduper = Deduplicator(CurationConfig(dedup_threshold=0.85))
    kept = deduper.dedupe(chunks)
    assert [c.id for c in kept] == ["a-0001", "a-0002", "a-0003"]


def test_dedup_near_duplicates_collapse_at_low_threshold():
    # Two variants of the same sentence — identical k=2 shingle overlap is
    # very high, so even a 0.6 threshold collapses them. Distinct content
    # below should remain.
    base = (
        "The contractor must keep accurate records of every variation order "
        "issued on the construction site at all times for every project they run."
    )
    near = (
        "The contractor must keep accurate records of every variation order "
        "issued on the construction site at all times for every project."
    )
    chunks = [
        _chunk("a-0001", base),
        _chunk("a-0002", near),
    ]
    deduper = Deduplicator(CurationConfig(dedup_threshold=0.6), shingle_size=2, num_perm=128)
    kept = deduper.dedupe(chunks)
    assert len(kept) == 1
    assert kept[0].id == "a-0001"


def test_dedup_preserves_order():
    chunks = [
        _chunk("z-0001", "alpha bravo charlie delta echo foxtrot"),
        _chunk("z-0002", "golf hotel india juliet kilo lima"),
        _chunk("z-0003", "mike november oscar papa quebec romeo"),
    ]
    deduper = Deduplicator(CurationConfig(dedup_threshold=0.85))
    kept = deduper.dedupe(chunks)
    assert [c.id for c in kept] == ["z-0001", "z-0002", "z-0003"]


def test_dedup_empty_input():
    deduper = Deduplicator(CurationConfig())
    assert deduper.dedupe([]) == []


def test_dedup_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        Deduplicator(CurationConfig(dedup_threshold=0.0))
    with pytest.raises(ValueError):
        Deduplicator(CurationConfig(dedup_threshold=1.5))
