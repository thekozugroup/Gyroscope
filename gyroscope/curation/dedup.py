"""Near-duplicate chunk removal using MinHash + LSH.

We shingle each chunk into overlapping word k-grams, build a MinHash, and
look up neighbours in an MinHashLSH index parameterised by the configured
Jaccard threshold. The first occurrence of any near-duplicate cluster
wins and its id is preserved.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from datasketch import MinHash, MinHashLSH

from gyroscope.core.config import CurationConfig
from gyroscope.core.models import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _shingles(tokens: list[str], k: int) -> set[str]:
    """Overlapping k-gram shingles. For inputs shorter than k we fall back
    to the unigram set so very short chunks still compare meaningfully."""
    if len(tokens) < k:
        return set(tokens)
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _minhash(shingles: Iterable[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(s.encode("utf-8"))
    return mh


class Deduplicator:
    """MinHash-LSH near-duplicate filter for Chunks."""

    def __init__(
        self,
        config: CurationConfig,
        *,
        shingle_size: int = 5,
        num_perm: int = 128,
    ) -> None:
        if not 0.0 < config.dedup_threshold <= 1.0:
            raise ValueError(
                f"dedup_threshold must be in (0, 1]; got {config.dedup_threshold}"
            )
        self._threshold = config.dedup_threshold
        self._shingle_size = shingle_size
        self._num_perm = num_perm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dedupe(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks with near-duplicates removed (first-wins)."""
        if not chunks:
            return []

        lsh = MinHashLSH(threshold=self._threshold, num_perm=self._num_perm)
        kept: list[Chunk] = []

        for chunk in chunks:
            mh = self._minhash_for(chunk)
            # Empty MinHash (no tokens) — keep the chunk only if it carries
            # text content the caller might want. It cannot be matched
            # against the LSH so we keep it as-is without indexing.
            if not chunk.text.strip():
                kept.append(chunk)
                continue

            neighbours = lsh.query(mh)
            if neighbours:
                # Near-duplicate of an already-kept chunk — drop it.
                continue
            lsh.insert(chunk.id, mh)
            kept.append(chunk)

        return kept

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _minhash_for(self, chunk: Chunk) -> MinHash:
        tokens = _tokenize(chunk.text)
        shingles = _shingles(tokens, self._shingle_size)
        return _minhash(shingles, self._num_perm)
