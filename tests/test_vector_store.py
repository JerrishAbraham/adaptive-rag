"""
Module 1, Step 6 — validation test for FAISS index build/save/load/search.

This is the Module 1 completion test: builds the full offline knowledge
base pipeline end-to-end (load -> preprocess -> chunk -> embed -> index),
persists it, reloads it fresh, and manually inspects retrieval results.
"""

import logging
import sys
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
    load_embedding_model,
)
from src.vector_store import build_faiss_index, save_index, load_index, search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TEST_INDEX_DIR = PROJECT_ROOT / "indexes" / "test_index"


def test_build_save_load_search():
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

    embedding_cfg = load_embedding_config(str(config_path))
    model_name = embedding_cfg["model_name"]
    normalize = embedding_cfg["normalize_embeddings"]

    embeddings = embed_chunks(chunks, model_name=model_name, normalize=normalize)
    matrix = embeddings_to_matrix(chunks, embeddings)

    # --- Build and save ---
    index = build_faiss_index(matrix)
    assert index.ntotal == len(chunks)

    save_index(index, chunks, TEST_INDEX_DIR)
    assert (TEST_INDEX_DIR / "index.faiss").exists()
    assert (TEST_INDEX_DIR / "metadata.json").exists()

    # --- Reload fresh (simulates a new process using the persisted index) ---
    reloaded_index, reloaded_metadata = load_index(TEST_INDEX_DIR)
    assert reloaded_index.ntotal == len(chunks)
    assert len(reloaded_metadata) == len(chunks)

    # --- Manual retrieval inspection using a real HotpotQA question ---
    sample_question = questions[0]
    print(f"\n=== Sample query ===\n{sample_question.question}")
    print(f"Gold answer: {sample_question.answer}")
    print(f"Supporting fact titles: {[t for t, _ in sample_question.supporting_facts]}")

    model = load_embedding_model(model_name)
    query_vector = model.encode(
        [sample_question.question], normalize_embeddings=normalize, convert_to_numpy=True
    )[0]

    results = search(reloaded_index, reloaded_metadata, query_vector, k=5)
    assert len(results) == 5

    print("\n=== Top-5 retrieved chunks ===")
    for r in results:
        print(f"rank={r['rank']} score={r['similarity_score']:.4f} "
              f"title={r['title']!r}")
        print(f"  text: {r['text'][:150]!r}...")

    # Sanity: scores should be in descending order.
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by descending similarity"

    # Sanity: at least one retrieved title should be a real article title
    # from this tiny corpus's supporting facts, most of the time.
    supporting_titles = {t for t, _ in sample_question.supporting_facts}
    retrieved_titles = {r["title"] for r in results}
    overlap = supporting_titles & retrieved_titles
    print(f"\nOverlap with gold supporting-fact titles: {overlap or 'none in top-5'}")

    print("\nOK: vector store test passed (Module 1 complete).")


if __name__ == "__main__":
    test_build_save_load_search()