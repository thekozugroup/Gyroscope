"""End-to-end curation pipeline.

Chunk -> dedup -> parallel extract -> synthesize -> write to disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from gyroscope.core.config import GyroscopeConfig
from gyroscope.core.llm import LLMClient
from gyroscope.core.models import Chunk, Document, GoldenDocument
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
from gyroscope.curation.synthesizer import ExtractorOutputs, synthesize

logger = logging.getLogger(__name__)


class CurationPipeline:
    """Distil a corpus of Documents into a single GoldenDocument."""

    def __init__(self, config: GyroscopeConfig) -> None:
        self._config = config

    async def distill(
        self,
        documents: list[Document],
        client: LLMClient,
    ) -> GoldenDocument:
        """Run the full curation pipeline. Writes ``golden.md`` and
        ``golden.json`` to ``config.output_dir`` before returning."""
        if not documents:
            raise ValueError("CurationPipeline.distill called with no documents.")

        curation_cfg = self._config.curation
        logger.info("Chunking %d documents", len(documents))
        chunker = SemanticChunker(curation_cfg)
        chunks: list[Chunk] = chunker.chunk_documents(documents)
        logger.info("Produced %d raw chunks", len(chunks))

        logger.info("Deduplicating chunks (threshold=%.2f)", curation_cfg.dedup_threshold)
        deduper = Deduplicator(curation_cfg)
        chunks = deduper.dedupe(chunks)
        logger.info("Retained %d chunks after dedup", len(chunks))

        # Run the six extractors in parallel; the LLMClient semaphore caps
        # concurrent in-flight requests.
        logger.info("Extracting identity / principles / procedures / knowledge / vocabulary / anti-patterns")
        identity, principles, procedures, knowledge, vocabulary, anti_patterns = (
            await asyncio.gather(
                extract_identity(chunks, client),
                extract_principles(chunks, client),
                extract_procedures(chunks, client),
                extract_knowledge(chunks, client),
                extract_vocabulary(chunks, client),
                extract_anti_patterns(chunks, client),
            )
        )

        extracts = ExtractorOutputs(
            identity=identity,
            principles=principles,
            procedures=procedures,
            knowledge=knowledge,
            vocabulary=vocabulary,
            anti_patterns=anti_patterns,
            source_documents=[d.source for d in documents],
        )

        logger.info("Synthesizing GoldenDocument")
        golden = await synthesize(extracts, client, config=curation_cfg)

        self._write_outputs(golden)
        return golden

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def _write_outputs(self, golden: GoldenDocument) -> None:
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        md_path = output_dir / "golden.md"
        json_path = output_dir / "golden.json"

        md_path.write_text(golden.to_markdown(), encoding="utf-8")
        json_path.write_text(
            json.dumps(golden.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Wrote %s and %s", md_path, json_path)
