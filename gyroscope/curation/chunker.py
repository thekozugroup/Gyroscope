"""Semantic chunking of Documents.

The chunker greedily packs paragraphs (then sentences inside an oversized
paragraph) into Chunks whose token count is close to
``CurationConfig.chunk_target_tokens``. It never splits inside a sentence and
prefers paragraph / heading boundaries.

Tokenizer
---------
We use ``tiktoken`` (``cl100k_base``) when available — it is a good cheap
proxy for the actual Anthropic tokenizer. When tiktoken is not available we
fall back to a ``len(text) // 4`` heuristic so the chunker still works in
environments without the optional dep.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from gyroscope.core.config import CurationConfig
from gyroscope.core.models import Chunk, Document

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _heuristic_token_count(text: str) -> int:
    """Cheap ``len(text) // 4`` fallback. Always returns at least 1 for
    non-empty input."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _build_token_counter() -> Callable[[str], int]:
    """Return a callable that counts tokens. Uses tiktoken if installed,
    otherwise the heuristic."""
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - optional dep
        return _heuristic_token_count

    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - network-free fallback
        return _heuristic_token_count

    def _count(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text, disallowed_special=()))

    return _count


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

# A "block" is a paragraph or heading line. Blocks are separated by one or
# more blank lines.
_BLOCK_SEP_RE = re.compile(r"\n\s*\n+")

# Sentence splitter — purposely simple. Splits on ``.``, ``!``, ``?`` followed
# by whitespace + capital/quote/paren, and on newlines that follow sentence
# punctuation. Never splits inside common abbreviations like "e.g." because
# they are not followed by whitespace+capital.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def _split_blocks(text: str) -> list[str]:
    """Split text into paragraph/heading blocks, preserving order, dropping
    purely empty blocks."""
    blocks = _BLOCK_SEP_RE.split(text.strip())
    return [b.strip() for b in blocks if b.strip()]


def _split_sentences(block: str) -> list[str]:
    """Split a single block into sentences. Falls back to the whole block
    when no sentence boundary is found."""
    parts = _SENT_SPLIT_RE.split(block.strip())
    return [p.strip() for p in parts if p.strip()]


def _source_hash(document: Document) -> str:
    """Stable short hash for a document used in chunk ids."""
    h = hashlib.sha1(document.source.encode("utf-8")).hexdigest()
    return h[:10]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


@dataclass
class _Pending:
    pieces: list[str]
    tokens: int

    def reset(self) -> None:
        self.pieces = []
        self.tokens = 0


class SemanticChunker:
    """Greedy paragraph/sentence chunker producing Chunks near the target
    token count.

    Parameters
    ----------
    config:
        Curation config holding ``chunk_target_tokens``.
    token_counter:
        Optional override; for tests we may inject a deterministic counter.
    """

    def __init__(
        self,
        config: CurationConfig,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._config = config
        self._count = token_counter or _build_token_counter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a single Document into ordered Chunks."""
        target = max(1, self._config.chunk_target_tokens)
        src_hash = _source_hash(document)
        chunks: list[Chunk] = []
        order = 0

        pending = _Pending(pieces=[], tokens=0)

        def flush() -> None:
            nonlocal order
            if not pending.pieces:
                return
            text = "\n\n".join(pending.pieces).strip()
            if text:
                chunk_id = f"{src_hash}-{order:04d}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_source=document.source,
                        text=text,
                        order=order,
                        metadata={"token_estimate": pending.tokens},
                    )
                )
                order += 1
            pending.reset()

        for block in _split_blocks(document.text):
            block_tokens = self._count(block)

            # A block on its own is bigger than the target -> sentence-split.
            if block_tokens > target:
                # Flush whatever we have first so the big block starts fresh.
                flush()
                self._emit_oversized_block(
                    block=block,
                    target=target,
                    document=document,
                    src_hash=src_hash,
                    chunks=chunks,
                    order_ref=[order],
                )
                # _emit_oversized_block mutates order via list reference.
                order = len(chunks)
                continue

            # If adding this block would overshoot the target, flush first
            # — but only if we already have content; a single under-target
            # block must always be allowed in.
            if pending.tokens and pending.tokens + block_tokens > target:
                flush()

            pending.pieces.append(block)
            pending.tokens += block_tokens

        flush()
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk many Documents preserving the input order."""
        out: list[Chunk] = []
        for doc in documents:
            out.extend(self.chunk_document(doc))
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit_oversized_block(
        self,
        *,
        block: str,
        target: int,
        document: Document,
        src_hash: str,
        chunks: list[Chunk],
        order_ref: list[int],
    ) -> None:
        """Split a single oversized block into sentence-packed chunks."""
        sentences = _split_sentences(block)
        if not sentences:
            return

        order = order_ref[0]
        buffer: list[str] = []
        buf_tokens = 0

        def flush_buffer() -> None:
            nonlocal order, buf_tokens
            if not buffer:
                return
            text = " ".join(buffer).strip()
            chunk_id = f"{src_hash}-{order:04d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_source=document.source,
                    text=text,
                    order=order,
                    metadata={"token_estimate": buf_tokens, "oversized_block": True},
                )
            )
            order += 1
            buffer.clear()
            buf_tokens = 0

        for sent in sentences:
            sent_tokens = self._count(sent)
            if buf_tokens and buf_tokens + sent_tokens > target:
                flush_buffer()
            buffer.append(sent)
            buf_tokens += sent_tokens

        flush_buffer()
        order_ref[0] = order
