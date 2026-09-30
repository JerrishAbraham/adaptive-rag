"""
Step 14 — 100-Question Pilot Run.

Rebuilds the corpus/index from 100 HotpotQA questions and runs the
UNCHANGED adaptive pipeline and fixed Top-5 baseline via the existing
evaluation harness. This script contains NO new retrieval, quality, or
generation logic — it only scales up sample size and aggregates/reports
results in more detail (K distribution, expansion frequency) than the
earlier smoke test did.

Explicitly out of scope for this run, per instruction:
  - No threshold tuning
  - No modification to adaptive_controller.py / context_quality.py /
    diversity_filter.py
  - No git commit/push

Results are saved to results/pilot_100_<timestamp>.json for later
reference, in addition to being printed.
"""

import json
import logging
import sys
import time
from collections import Counter
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
logger = logging.getLogger(__name__)

SAMPLE_SIZE = 100
INDEX_DIR = PROJECT_ROOT / "indexes" / "pilot_100_index"
RESULTS_DIR = PROJECT_ROOT / "results"


def build_pilot_index():
    logger.info("=== Step 14 Pilot: loading %d HotpotQA questions ===", SAMPLE_SIZE)
    documents, questions = load_hotpotqa_sample(
        split="validation", sample_limit=SAMPLE_SIZE, config="distractor"
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
    save_index(index, chunks, INDEX_DIR)
    index, metadata = load_index(INDEX_DIR)
    chunk_embeddings = {c.chunk_id: embeddings[c.chunk_id] for c in chunks}

    logger.info(
        "Pilot corpus ready: %d documents -> %d chunks -> %d indexed vectors",
        len(cleaned_docs), len(chunks), index.ntotal,
    )
    return index, metadata, chunk_embeddings, questions


def compute_k_distribution(summary) -> dict:
    """Count how often each K value was used across the question set."""
    counts = Counter(r.k_used for r in summary.per_question)
    return dict(sorted(counts.items()))


def compute_expansion_frequency(summary) -> dict:
    """
    Fraction of questions where the adaptive loop expanded at least once
    (num_iterations > 1), plus the raw count. Always 0 for baseline by
    construction (not called for baseline in main()).
    """
    n = summary.num_questions
    expanded = sum(1 for r in summary.per_question if r.num_iterations > 1)
    return {
        "expanded_count": expanded,
        "expanded_fraction": round(expanded / n, 4) if n else 0.0,
        "total_questions": n,
    }


def summary_to_dict(summary) -> dict:
    return {
        "num_questions": summary.num_questions,
        "avg_em": round(summary.avg_em, 4),
        "avg_f1": round(summary.avg_f1, 4),
        "avg_supporting_fact_recall": round(summary.avg_supporting_fact_recall, 4),
        "avg_k": round(summary.avg_k, 4),
        "avg_iterations": round(summary.avg_iterations, 4),
        "avg_latency_seconds": round(summary.avg_latency_seconds, 4),
        "generation_success_rate": round(summary.generation_success_rate, 4),
    }


def per_question_to_dicts(summary) -> list[dict]:
    return [
        {
            "question_id": r.question_id,
            "question": r.question,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "em": r.em,
            "f1": round(r.f1, 4),
            "supporting_fact_recall": round(r.supporting_fact_recall, 4),
            "k_used": r.k_used,
            "num_iterations": r.num_iterations,
            "latency_seconds": r.latency_seconds,
            "generation_success": r.generation_success,
            "confidence_score": r.confidence_score,
            "sufficient": r.sufficient,
            "quality_metrics": r.quality_metrics,
        }
        for r in summary.per_question
    ]


def main():
    start = time.time()
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    index, metadata, chunk_embeddings, questions = build_pilot_index()

    adaptive_pipeline = AdaptiveRAGPipeline(index, metadata, chunk_embeddings, config_path=str(config_path))
    baseline_pipeline = FixedTopKPipeline(index, metadata, baseline_k=5, config_path=str(config_path))

    logger.info("=== Running adaptive pipeline on %d questions (unchanged logic) ===", len(questions))
    adaptive_summary = evaluate_adaptive_pipeline(adaptive_pipeline, questions)

    logger.info("=== Running fixed Top-5 baseline on %d questions (unchanged logic) ===", len(questions))
    baseline_summary = evaluate_baseline_pipeline(baseline_pipeline, questions)

    total_runtime = time.time() - start

    k_dist_adaptive = compute_k_distribution(adaptive_summary)
    k_dist_baseline = compute_k_distribution(baseline_summary)
    expansion_stats = compute_expansion_frequency(adaptive_summary)

    # --- Console report ---
    print("\n" + "=" * 70)
    print(f"STEP 14 PILOT RESULTS — {len(questions)} questions, HotpotQA distractor/validation")
    print("=" * 70)

    print_comparison(adaptive_summary, baseline_summary)

    print("\n=== K Distribution (Adaptive) ===")
    for k, count in k_dist_adaptive.items():
        pct = 100 * count / adaptive_summary.num_questions
        print(f"  K={k}: {count} questions ({pct:.1f}%)")

    print("\n=== K Distribution (Baseline, always 5 by construction) ===")
    for k, count in k_dist_baseline.items():
        print(f"  K={k}: {count} questions")

    print("\n=== Expansion Frequency (Adaptive) ===")
    print(f"  Questions where the loop expanded beyond initial K: "
          f"{expansion_stats['expanded_count']}/{expansion_stats['total_questions']} "
          f"({expansion_stats['expanded_fraction']:.1%})")

    print(f"\nTotal pilot runtime: {total_runtime:.1f}s "
          f"({total_runtime / len(questions):.2f}s/question average, both pipelines combined)")

    # --- Per-question table ---
    print("\n=== Per-Question Results (Adaptive) ===")
    print(f"{'qid':<26}{'EM':<5}{'F1':<7}{'SF-recall':<11}{'K':<4}{'iters':<7}{'latency(s)':<10}")
    for r in adaptive_summary.per_question:
        print(f"{r.question_id:<26}{r.em:<5.0f}{r.f1:<7.2f}{r.supporting_fact_recall:<11.2f}"
              f"{r.k_used:<4}{r.num_iterations:<7}{r.latency_seconds:<10.2f}")

    # --- Save full results to disk ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"pilot_100_{timestamp}.json"

    output = {
        "run_type": "pilot_100_questions",
        "sample_size": len(questions),
        "notes": "No threshold tuning or logic changes applied. Pilot run only, per Step 14 scope.",
        "adaptive_summary": summary_to_dict(adaptive_summary),
        "baseline_summary": summary_to_dict(baseline_summary),
        "k_distribution_adaptive": k_dist_adaptive,
        "k_distribution_baseline": k_dist_baseline,
        "expansion_frequency": expansion_stats,
        "total_runtime_seconds": round(total_runtime, 2),
        "adaptive_per_question": per_question_to_dicts(adaptive_summary),
        "baseline_per_question": per_question_to_dicts(baseline_summary),
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: {results_path}")
    print("\nOK: Step 14 pilot run complete.")


if __name__ == "__main__":
    main()