"""
Module 1, Step 4/5 — validation test for BGE embedding generation and caching.

Loads a tiny HotpotQA sample, preprocesses and chunks it, generates
embeddings, and verifies shape, normalization, and that the disk cache
is actually used on a second call (no recomputation).
"""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_hotpotqa_sample  # noqa: E402
from src.preprocessing import preprocess_documents  # noqa: E402
from src.chunking import chunk_documents, load_chunking_config  # noqa: E402
from src.embeddings import (  # noqa: E402
    embed_chunks,
    embeddings_to_matrix,
    load_embedding_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TEST_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "test_embedding_cache.pkl"


def test_embed_tiny_sample():
    documents, questions = load_hotpotqa_sample(
        split="validation", sample_limit=10, config="distractor"
    )
    cleaned_docs = preprocess_documents(documents)

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    chunking_cfg = load_chunking_config(str(config_path))
    chunks = chunk_documents(
        cleaned_docs,
        chunk_size=chunking_cfg["chunk_size"],
        chunk_overlap=chunking_cfg["chunk_overlap"],
    )
    print(f"\nChunks to embed: {len(chunks)}")

    embedding_cfg = load_embedding_config(str(config_path))
    model_name = embedding_cfg["model_name"]
    normalize = embedding_cfg["normalize_embeddings"]

    # Clean slate for this test's cache file.
    if TEST_CACHE_PATH.exists():
        TEST_CACHE_PATH.unlink()

    # --- First call: everything should be computed fresh. ---
    t0 = time.time()
    embeddings = embed_chunks(
        chunks, model_name=model_name, normalize=normalize, cache_path=TEST_CACHE_PATH
    )
    first_call_seconds = time.time() - t0

    assert len(embeddings) == len(chunks)
    matrix = embeddings_to_matrix(chunks, embeddings)
    print(f"Embedding matrix shape: {matrix.shape}")

    dim = matrix.shape[1]
    assert dim in (768,), f"Unexpected embedding dimension {dim} for bge-base-en-v1.5"

    if normalize:
        norms = (matrix ** 2).sum(axis=1) ** 0.5
        max_deviation = abs(norms - 1.0).max()
        print(f"Max deviation from unit norm: {max_deviation:.6f}")
        assert max_deviation < 1e-3, "Embeddings are not properly normalized"

    # --- Second call: should hit the cache, not recompute. ---
    t1 = time.time()
    embeddings_again = embed_chunks(
        chunks, model_name=model_name, normalize=normalize, cache_path=TEST_CACHE_PATH
    )
    second_call_seconds = time.time() - t1

    assert set(embeddings_again.keys()) == set(embeddings.keys())
    import numpy as np
    for chunk_id in embeddings:
        assert np.allclose(embeddings[chunk_id], embeddings_again[chunk_id])

    print(f"\nFirst call (compute): {first_call_seconds:.2f}s")
    print(f"Second call (cached): {second_call_seconds:.2f}s")
    assert second_call_seconds < first_call_seconds, (
        "Second call was not faster — cache may not be working."
    )

    print("\nOK: embeddings test passed.")

    # Clean up test cache file.
    if TEST_CACHE_PATH.exists():
        TEST_CACHE_PATH.unlink()


if __name__ == "__main__":
    test_embed_tiny_sample()