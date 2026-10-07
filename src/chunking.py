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
    sentences = document.sentences
    if not sentences:
        return []

    chunks: list[ChunkRecord] = []
    buffer: list[str] = []
    buffer_len = 0
    chunk_index = 0

    def flush_chunk(next_buffer_seed: list[str]) -> list[str]:
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
        buffer_len += len(sentence) + 1

        if buffer_len >= chunk_size:
            overlap_seed: list[str] = []
            overlap_len = 0

            for s in reversed(buffer):
                if overlap_len >= chunk_overlap:
                    break
                overlap_seed.insert(0, s)
                overlap_len += len(s) + 1

            if len(overlap_seed) >= len(buffer):
                overlap_seed = []

            buffer = flush_chunk(overlap_seed)
            buffer_len = sum(len(s) + 1 for s in buffer)

    if buffer:
        flush_chunk([])

    logger.debug(
        "Document %s (%s): %d sentences -> %d chunks",
        document.document_id,
        document.title,
        len(sentences),
        len(chunks),
    )

    return chunks


def chunk_documents(
    documents: list[DocumentRecord],
    chunk_size: int = 400,
    chunk_overlap: int = 75,
) -> list[ChunkRecord]:
    all_chunks: list[ChunkRecord] = []

    for doc in documents:
        all_chunks.extend(
            chunk_document(
                doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    logger.info(
        "Chunked %d documents into %d chunks (chunk_size=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )

    return all_chunks


def load_chunking_config(config_path: str = "configs/config.yaml") -> dict:
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["chunking"]