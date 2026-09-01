"""
Module 3 — validation test for context quality assessment, the bounded
adaptive loop, and semantic diversity filtering.

Builds a small FAISS index, runs real HotpotQA questions through initial
retrieval (Module 2), then the adaptive loop (Module 3 Steps 11-12), then
diversity filtering (Step 13), inspecting behavior at each stage.
"""

import logging
import sys
from pathlib import Path

import numpy as np

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
from src.adaptive_controller import run_initial_retrieval, load_adaptive_controller_config  # noqa: E402
from src.context_quality import run_adaptive_loop, load_context_quality_config  # noqa: E402
from src.diversity_filter import apply_diversity_filter, load_diversity_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TEST_INDEX_DIR = PROJECT_ROOT / "indexes" / "test_index_module3"


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
    return load_index(TEST_INDEX_DIR), questions, chunks, embeddings


def test_full_module3_pipeline():
    (index, metadata), questions, chunks, chunk_embeddings = _build_test_index()

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    qa_config = load_query_analysis_config(str(config_path))
    ac_config = load_adaptive_controller_config(str(config_path))
    quality_config = load_context_quality_config(str(config_path))
    diversity_config = load_diversity_config(str(config_path))

    import yaml
    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    k_values = full_config["retrieval"]["adaptive_k_values"]
    max_k = full_config["retrieval"]["max_k"]
    max_iterations = full_config["retrieval"]["max_iterations"]

    embedding_cfg = load_embedding_config(str(config_path))
    model = load_embedding_model(embedding_cfg["model_name"])

    print(f"\nk_values={k_values} max_k={max_k} max_iterations={max_iterations}")
    print(f"quality confidence_threshold={quality_config['confidence_threshold']}")
    print(f"diversity lambda={diversity_config['lambda']} target_size={diversity_config['target_context_size']}")

    for q in questions[:5]:
        profile = analyze_query(
            q.question, qa_config, embedding_model=model,
            normalize_embedding=embedding_cfg["normalize_embeddings"],
        )
        initial_results, initial_decision = run_initial_retrieval(
            profile, index, metadata, ac_config, k_values
        )

        final_results, quality_history, decision_history = run_adaptive_loop(
            profile, index, metadata, initial_results, initial_decision,
            quality_config, k_values, max_k, max_iterations,
        )

        # --- Assertions on the loop itself ---
        assert len(quality_history) >= 1
        assert len(decision_history) >= 2  # initial_select + at least one terminal action
        last_action = decision_history[-1].action
        assert last_action in ("accept", "stop_max_k", "stop_max_iterations")
        # Loop must never exceed max_iterations expansion steps.
        expand_count = sum(1 for d in decision_history if d.action == "expand")
        assert expand_count < max_iterations

        print(f"\nQuestion: {q.question}")
        print(f"  intent={profile.intent} complexity={profile.complexity_score}")
        for d in decision_history:
            print(f"    [{d.action}] iter={d.iteration} k={d.current_k}->{d.next_k}: {d.reason}")
        print(f"  final result count: {len(final_results)}")

        # --- Diversity filtering ---
        candidate_embeddings = {c.chunk_id: chunk_embeddings[c.chunk_id] for c in chunks}
        filtered_results, filter_stats = apply_diversity_filter(
            final_results, candidate_embeddings, profile.embedding,
            lambda_param=diversity_config["lambda"],
            target_size=diversity_config["target_context_size"],
        )

        assert filter_stats["post_filter_count"] <= filter_stats["pre_filter_count"]
        assert filter_stats["post_filter_count"] == len(filtered_results)
        assert len(filter_stats["selected_chunk_ids"]) == len(set(filter_stats["selected_chunk_ids"])), (
            "Diversity filter selected a duplicate chunk_id."
        )

        print(f"  diversity filter: {filter_stats['pre_filter_count']} -> "
              f"{filter_stats['post_filter_count']} chunks")
        for r in filtered_results:
            print(f"    kept: title={r['title']!r} score={r['similarity_score']:.4f} "
                  f"chunk_id={r['chunk_id']}")

    print("\nOK: Module 3 (quality assessment, adaptive loop, diversity filtering) test passed.")


if __name__ == "__main__":
    test_full_module3_pipeline()