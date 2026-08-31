"""
Module 1, Step 6 — FAISS Index Construction and Persistence.

Builds a FAISS index from chunk embeddings and persists it alongside a
separate chunk metadata mapping (FAISS position -> ChunkRecord info).
FAISS itself only stores vectors and integer positions; it knows
nothing about chunk_id/document_id/title/text, so that mapping must be
maintained ourselves and kept in exact sync with index insertion order.

Metric: cosine similarity, implemented as inner product (IndexFlatIP)
over L2-normalized vectors (normalization already done in embeddings.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.chunking import ChunkRecord

logger = logging.getLogger(__name__)


def build_faiss_index(embedding_matrix: np.ndarray):
    """
    Build a flat FAISS index (exact search, no approximation) using inner
    product similarity. Since vectors are L2-normalized upstream, inner
    product is equivalent to cosine similarity.

    A flat index is intentional at this project's scale (a HotpotQA
    subset, not a production-scale corpus) — exact search, no ANN
    approximation error to reason about during evaluation.
    """
    import faiss

    if embedding_matrix.dtype != np.float32:
        embedding_matrix = embedding_matrix.astype(np.float32)

    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embedding_matrix)

    logger.info("Built FAISS IndexFlatIP: dim=%d, ntotal=%d", dim, index.ntotal)
    return index


def save_index(index, chunks: list[ChunkRecord], index_dir: str | Path) -> None:
    """
    Persist the FAISS index and a parallel chunk metadata mapping.

    Files written:
        {index_dir}/index.faiss   - the FAISS index itself
        {index_dir}/metadata.json - list of chunk metadata, position i
                                     in this list corresponds exactly to
                                     FAISS internal id i (insertion order)
    """
    import faiss

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "index.faiss"))

    metadata = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "title": c.title,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    with open(index_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    logger.info(
        "Saved FAISS index and metadata for %d chunks to %s", len(chunks), index_dir
    )


def load_index(index_dir: str | Path):
    """
    Load a previously persisted FAISS index and its metadata mapping.

    Returns:
        (index, metadata) where metadata is a list of dicts, and
        metadata[i] corresponds exactly to FAISS internal id i.
    """
    import faiss

    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))

    with open(index_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata mismatch: FAISS has {index.ntotal} vectors "
            f"but metadata.json has {len(metadata)} entries. "
            f"The index directory may be corrupted or partially written."
        )

    logger.info("Loaded FAISS index and metadata: %d vectors", index.ntotal)
    return index, metadata


def search(index, metadata: list[dict], query_vector: np.ndarray, k: int) -> list[dict]:
    """
    Search the index for the top-k nearest chunks to a query vector.

    Args:
        query_vector: a single L2-normalized embedding, shape (dim,) or (1, dim).
        k: number of results to return.

    Returns:
        List of dicts (length <= k, sorted by descending similarity), each
        containing chunk_id, document_id, title, text, rank, and similarity_score.
        This shape matches the RetrievalResult structure from the reference doc.
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    query_vector = query_vector.astype(np.float32)

    k = min(k, index.ntotal)
    scores, ids = index.search(query_vector, k)

    results = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):
        if idx == -1:  # FAISS pads with -1 if fewer than k results exist
            continue
        entry = metadata[idx]
        results.append(
            {
                "chunk_id": entry["chunk_id"],
                "document_id": entry["document_id"],
                "title": entry["title"],
                "text": entry["text"],
                "rank": rank,
                "similarity_score": float(score),
            }
        )
    return results