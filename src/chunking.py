"""
Module 1, Step 3 — Sentence-Aware Chunking and Metadata Creation.

Splits DocumentRecords into ChunkRecords. Chunk boundaries always fall
on sentence boundaries (never mid-sentence) — a chunk is built by
accumulating whole sentences until a target character size is reached,
then the next chunk starts with an overlapping tail of prior sentences.

Short articles that fit within one chunk are kept as a single chunk,
per the implementation reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.data_loader import DocumentRecord

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    title: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_document(
    document: DocumentRecord,
    chunk_size: int = 400,
    chunk_overlap: int = 75,
) -> list[ChunkRecord]:
    """
    Split one document's sentences into overlapping, sentence-aligned chunks.

    Args:
        document: a preprocessed DocumentRecord (see preprocessing.py).
        chunk_size: target chunk size in characters (not tokens — kept
            simple and dependency-free at this stage).
        chunk_overlap: approximate character overlap carried into the
            start of the next chunk, to avoid losing context at boundaries.

    Returns:
        List of ChunkRecords in order, chunk_index starting at 0.

    Chunking algorithm:
        Walk through sentences, accumulating them into a growing buffer.
        Once the buffer's length reaches chunk_size, close the chunk.
        Start the next chunk by carrying over trailing sentences from the
        buffer whose combined length is >= chunk_overlap (whole sentences
        only, never partial), then keep accumulating from there.
    """
    sentences = document.sentences
    if not sentences:
        return []

    chunks: list[ChunkRecord] = []
    buffer: list[str] = []
    buffer_len = 0
    chunk_index = 0

    def flush_chunk(next_buffer_seed: list[str]) -> list[str]:
        """Close out the current buffer as a chunk, return the seed for the next buffer."""
        nonlocal chunk_index
        chunk_text = " ".join(buffer)
        chunk_id = f"{document.document_id}_chunk_{chunk_index:04d}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=document.document_id,
                title=document.title,
                chunk_index=chunk_index,
                text=chunk_text,
                metadata={
                    "source": document.source,
                    "num_sentences": len(buffer),
                    "char_length": len(chunk_text),
                },
            )
        )
        chunk_index += 1
        return next_buffer_seed

    for sentence in sentences:
        buffer.append(sentence)
        buffer_len += len(sentence) + 1  # +1 for the joining space

        if buffer_len >= chunk_size:
            # Determine overlap seed: walk back from the end of buffer,
            # collecting whole sentences until we reach chunk_overlap chars.
            overlap_seed: list[str] = []
            overlap_len = 0
            for s in reversed(buffer):
                if overlap_len >= chunk_overlap:
                    break
                overlap_seed.insert(0, s)
                overlap_len += len(s) + 1

            # Never let the overlap seed swallow the entire buffer
            # (would cause an infinite loop of identical chunks).
            if len(overlap_seed) >= len(buffer):
                overlap_seed = []

            buffer = flush_chunk(overlap_seed)
            buffer_len = sum(len(s) + 1 for s in buffer)

    # Flush any remaining sentences as the final chunk.
    if buffer:
        flush_chunk([])

    logger.debug(
        "Document %s (%s): %d sentences -> %d chunks",
        document.document_id, document.title, len(sentences), len(chunks),
    )
    return chunks


def chunk_documents(
    documents: list[DocumentRecord],
    chunk_size: int = 400,
    chunk_overlap: int = 75,
) -> list[ChunkRecord]:
    """Chunk a full list of documents and return a flat list of ChunkRecords."""
    all_chunks: list[ChunkRecord] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

    logger.info(
        "Chunked %d documents into %d chunks (chunk_size=%d, overlap=%d)",
        len(documents), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks


def load_chunking_config(config_path: str = "configs/config.yaml") -> dict:
    """Load chunk_size/chunk_overlap from the project config file."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["chunking"]