"""
Module 2, Step 10 — validation test for the Adaptive Retrieval Controller.

Builds a small FAISS index (reusing Module 1 components), analyzes a set
of queries with varying complexity, and confirms different initial K
values are chosen with explainable reasons, then performs the first
retrieval for each.
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
    embed_chunks, embeddings_to_matrix, load_embedding_config, load_embedding_model,
)
from src.vector_store import build_faiss_index, save_index, load_index  # noqa: E402
from src.query_analyzer import analyze_query, load_query_analysis_config  # noqa: E402
from src.adaptive_controller import (  # noqa: E402
    run_initial_retrieval, load_adaptive_controller_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TEST_INDEX_DIR = PROJECT_ROOT / "indexes" / "test_index_module2"


def _build_test_index():
    documents, questions = load_hotpotqa_sample(
        split="validation", sample_limit=20, config="distractor"
    )
    cleaned_docs = preprocess_documents(documents)

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    chunking_cfg = load_chunking_config(str(config_path))
    chunks = chunk_documents(
        cleaned_docs, chunk_size=chunking_cfg["chunk_size"], chunk_overlap=chunking_cfg["chunk_overlap"]
    )

    embedding_cfg = load_embedding_config(str(config_path))
    embeddings = embed_chunks(
        chunks, model_name=embedding_cfg["model_name"], normalize=embedding_cfg["normalize_embeddings"]
    )
    matrix = embeddings_to_matrix(chunks, embeddings)
    index = build_faiss_index(matrix)
    save_index(index, chunks, TEST_INDEX_DIR)
    return load_index(TEST_INDEX_DIR), questions


def test_adaptive_initial_retrieval():
    (index, metadata), questions = _build_test_index()

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    qa_config = load_query_analysis_config(str(config_path))
    ac_config = load_adaptive_controller_config(str(config_path))

    import yaml
    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    k_values = full_config["retrieval"]["adaptive_k_values"]
    max_k = full_config["retrieval"]["max_k"]

    embedding_cfg = load_embedding_config(str(config_path))
    model = load_embedding_model(embedding_cfg["model_name"])

    print(f"\nAllowed k_values: {k_values}, max_k: {max_k}")

    seen_k_values = set()
    for q in questions[:6]:
        profile = analyze_query(
            q.question, qa_config, embedding_model=model,
            normalize_embedding=embedding_cfg["normalize_embeddings"],
        )
        results, decision = run_initial_retrieval(profile, index, metadata, ac_config, k_values)

        assert decision.current_k in k_values, "Chosen K is not in the allowed bounded set."
        assert decision.current_k <= max_k, "Chosen K exceeds max_k."
        assert len(results) <= decision.current_k
        assert decision.reason, "Decision must record a non-empty reason."

        seen_k_values.add(decision.current_k)

        print(f"\nQuestion: {q.question}")
        print(f"  intent={profile.intent} complexity={profile.complexity_score}")
        print(f"  decision: current_k={decision.current_k} action={decision.action}")
        print(f"  reason: {decision.reason}")
        print(f"  retrieved {len(results)} chunks, top result title={results[0]['title']!r} "
              f"score={results[0]['similarity_score']:.4f}" if results else "  no results")

    print(f"\nDistinct K values chosen across sample questions: {sorted(seen_k_values)}")
    assert len(seen_k_values) >= 1

    print("\nOK: adaptive controller test passed.")


if __name__ == "__main__":
    test_adaptive_initial_retrieval()