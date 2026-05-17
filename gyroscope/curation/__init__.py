"""Phase 2 — Curation: distil a corpus of Documents into a GoldenDocument."""

from gyroscope.curation.chunker import SemanticChunker
from gyroscope.curation.dedup import Deduplicator
from gyroscope.curation.extractor import (
    extract_anti_patterns,
    extract_identity,
    extract_knowledge,
    extract_principles,
    extract_procedures,
    extract_vocabulary,
)
from gyroscope.curation.pipeline import CurationPipeline
from gyroscope.curation.synthesizer import ExtractorOutputs, synthesize

__all__ = [
    "CurationPipeline",
    "Deduplicator",
    "ExtractorOutputs",
    "SemanticChunker",
    "extract_anti_patterns",
    "extract_identity",
    "extract_knowledge",
    "extract_principles",
    "extract_procedures",
    "extract_vocabulary",
    "synthesize",
]
