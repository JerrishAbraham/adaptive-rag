from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.chunking import ChunkRecord

logger = logging.getLogger(__name__)


def build_faiss_index(embedding_matrix: np.ndarray):
    
    import faiss

    if embedding_matrix.dtype != np.float32:
        embedding_matrix = embedding_matrix.astype(np.float32)

    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embedding_matrix)

    logger.info("Built FAISS IndexFlatIP: dim=%d, ntotal=%d", dim, index.ntotal)
    return index


def save_index(index, chunks: list[ChunkRecord], index_dir: str | Path) -> None:
   
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