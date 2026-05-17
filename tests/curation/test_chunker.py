"""Tests for the SemanticChunker."""

from __future__ import annotations

from gyroscope.core.config import CurationConfig
from gyroscope.core.models import Document, DocumentKind
from gyroscope.curation.chunker import SemanticChunker


def _doc(text: str, source: str = "memo://doc-a") -> Document:
    return Document(source=source, kind=DocumentKind.TXT, text=text)


def _make_chunker(target: int = 30, counter=None) -> SemanticChunker:
    cfg = CurationConfig(chunk_target_tokens=target)
    return SemanticChunker(cfg, token_counter=counter or (lambda s: max(1, len(s.split()))))


def test_chunker_respects_paragraph_boundaries():
    text = "Para one sentence A. Para one sentence B.\n\nPara two text.\n\nPara three text here."
    chunker = _make_chunker(target=10)
    chunks = chunker.chunk_document(_doc(text))

    # Each chunk must be composed of whole paragraphs only.
    for chunk in chunks:
        for paragraph in chunk.text.split("\n\n"):
            assert paragraph.strip() in text


def test_chunker_packs_small_paragraphs_together():
    # Six tiny paragraphs, target = 6 word-tokens => roughly 3 chunks.
    text = "\n\n".join([f"para {i} alpha beta" for i in range(6)])
    chunker = _make_chunker(target=6)
    chunks = chunker.chunk_document(_doc(text))

    assert 1 < len(chunks) <= 6
    # No chunk should explode well beyond the target.
    for c in chunks:
        assert c.metadata["token_estimate"] <= 6 * 2  # comfortable headroom


def test_chunker_never_splits_mid_sentence_in_oversized_block():
    # Single paragraph well over target, multiple sentences.
    text = " ".join([f"This is sentence number {i}." for i in range(12)])
    chunker = _make_chunker(target=5)
    chunks = chunker.chunk_document(_doc(text))

    for chunk in chunks:
        # Each chunk text should end with a sentence terminator (the
        # sentence splitter keeps terminators on the prior segment).
        assert chunk.text.rstrip().endswith((".", "!", "?"))


def test_chunker_ids_are_stable_and_ordered():
    text = "A\n\nB\n\nC\n\nD"
    chunker = _make_chunker(target=1)
    chunks = chunker.chunk_document(_doc(text, source="memo://doc-x"))

    ids = [c.id for c in chunks]
    orders = [c.order for c in chunks]
    assert orders == list(range(len(chunks)))
    # Ids are zero-padded, deterministic, prefixed by a stable source hash.
    prefix = ids[0].rsplit("-", 1)[0]
    assert all(cid.startswith(prefix + "-") for cid in ids)
    suffixes = [int(cid.rsplit("-", 1)[1]) for cid in ids]
    assert suffixes == list(range(len(chunks)))

    # Rerun -> identical ids.
    chunker2 = _make_chunker(target=1)
    chunks2 = chunker2.chunk_document(_doc(text, source="memo://doc-x"))
    assert [c.id for c in chunks2] == ids


def test_chunker_handles_empty_document():
    chunker = _make_chunker(target=10)
    assert chunker.chunk_document(_doc("")) == []
    assert chunker.chunk_document(_doc("   \n\n  \n")) == []


def test_chunker_multiple_documents_preserve_order():
    chunker = _make_chunker(target=10)
    docs = [
        _doc("alpha beta gamma", source="src://a"),
        _doc("delta epsilon", source="src://b"),
    ]
    chunks = chunker.chunk_documents(docs)
    assert len(chunks) == 2
    assert chunks[0].document_source == "src://a"
    assert chunks[1].document_source == "src://b"
    # Each document has its own id namespace.
    assert chunks[0].id != chunks[1].id


def test_oversized_block_chunks_marked_in_metadata():
    text = " ".join([f"Sentence {i} body words." for i in range(15)])
    chunker = _make_chunker(target=5)
    chunks = chunker.chunk_document(_doc(text))
    assert any(c.metadata.get("oversized_block") for c in chunks)
