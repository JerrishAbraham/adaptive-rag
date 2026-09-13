"""
Module 5, Part B — validation test for the evaluation harness.

Builds a shared index, runs both the adaptive pipeline and the fixed
Top-5 baseline over the same small question set, computes metrics for
each, and prints the side-by-side comparison.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_hotpotqa_sample  # noqa: E402
from src.preprocessing import preprocess_documents  # noqa: E402
from src.chunking import chunk_documents, load_chunking_config  # noqa: E402
from src.embeddings import embed_chunks, embeddings_to_matrix, load_embedding_config  # noqa: E402
from src.vector_store import build_faiss_index, save_index, load_index  # noqa: E402
from src.pipeline import AdaptiveRAGPipeline  # noqa: E402
from src.baseline_pipeline import FixedTopKPipeline  # noqa: E402
from src.evaluation import evaluate_adaptive_pipeline, evaluate_baseline_pipeline, print_comparison  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TEST_INDEX_DIR = PROJECT_ROOT / "indexes" / "test_index_eval"


def test_evaluation_harness():
    documents, questions = load_hotpotqa_sample(split="validation", sample_limit=15, config="distractor")
    cleaned_docs = preprocess_documents(documents)

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    chunking_cfg = load_chunking_config(str(config_path))
    chunks = chunk_documents(cleaned_docs, chunk_size=chunking_cfg["chunk_size"], chunk_overlap=chunking_cfg["chunk_overlap"])

    embedding_cfg = load_embedding_config(str(config_path))
    embeddings = embed_chunks(chunks, model_name=embedding_cfg["model_name"], normalize=embedding_cfg["normalize_embeddings"])
    matrix = embeddings_to_matrix(chunks, embeddings)
    index = build_faiss_index(matrix)
    save_index(index, chunks, TEST_INDEX_DIR)
    index, metadata = load_index(TEST_INDEX_DIR)
    chunk_embeddings = {c.chunk_id: embeddings[c.chunk_id] for c in chunks}

    eval_questions = questions[:5]

    adaptive_pipeline = AdaptiveRAGPipeline(index, metadata, chunk_embeddings, config_path=str(config_path))
    baseline_pipeline = FixedTopKPipeline(index, metadata, baseline_k=5, config_path=str(config_path))

    print(f"\nEvaluating {len(eval_questions)} questions with adaptive pipeline...")
    adaptive_summary = evaluate_adaptive_pipeline(adaptive_pipeline, eval_questions)

    print(f"\nEvaluating {len(eval_questions)} questions with baseline (Fixed Top-5) pipeline...")
    baseline_summary = evaluate_baseline_pipeline(baseline_pipeline, eval_questions)

    assert adaptive_summary.num_questions == len(eval_questions)
    assert baseline_summary.num_questions == len(eval_questions)
    assert 0.0 <= adaptive_summary.avg_em <= 1.0
    assert 0.0 <= adaptive_summary.avg_f1 <= 1.0
    assert baseline_summary.avg_k == 5.0

    print("\n=== Per-question detail (adaptive) ===")
    for r in adaptive_summary.per_question:
        print(f"  qid={r.question_id} em={r.em:.0f} f1={r.f1:.2f} sf_recall={r.supporting_fact_recall:.2f} "
              f"k={r.k_used} iters={r.num_iterations}")
        print(f"    Q: {r.question}")
        print(f"    gold: {r.gold_answer!r}  pred: {r.predicted_answer!r}")

    print_comparison(adaptive_summary, baseline_summary)

    print("\nOK: evaluation harness test passed.")


if __name__ == "__main__":
    test_evaluation_harness()