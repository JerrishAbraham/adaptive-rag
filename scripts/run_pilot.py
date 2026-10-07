import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_hotpotqa_sample  # noqa: E402
from src.preprocessing import preprocess_documents  # noqa: E402
from src.chunking import chunk_documents, load_chunking_config  # noqa: E402
from src.embeddings import embed_chunks, embeddings_to_matrix, load_embedding_config  # noqa: E402
from src.vector_store import build_faiss_index, save_index, load_index  # noqa: E402
from src.pipeline import AdaptiveRAGPipeline  # noqa: E402
from src.baseline_pipeline import FixedTopKPipeline  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_adaptive_pipeline,
    evaluate_baseline_pipeline,
    print_comparison,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# PILOT CONFIGURATION
# ============================================================

SAMPLE_SIZE = 30

INDEX_DIR = PROJECT_ROOT / "indexes" / f"pilot_{SAMPLE_SIZE}_index"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# INDEX BUILDING
# ============================================================

def build_pilot_index():
    logger.info(
        "=== Step 14 Pilot: loading %d HotpotQA questions ===",
        SAMPLE_SIZE,
    )

    documents, questions = load_hotpotqa_sample(
        split="validation",
        sample_limit=SAMPLE_SIZE,
        config="distractor",
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

    embeddings = embed_chunks(
        chunks,
        model_name=embedding_cfg["model_name"],
        normalize=embedding_cfg["normalize_embeddings"],
    )

    matrix = embeddings_to_matrix(chunks, embeddings)

    index = build_faiss_index(matrix)

    save_index(index, chunks, INDEX_DIR)

    index, metadata = load_index(INDEX_DIR)

    chunk_embeddings = {
        c.chunk_id: embeddings[c.chunk_id]
        for c in chunks
    }

    logger.info(
        "Pilot corpus ready: %d documents -> %d chunks -> %d indexed vectors",
        len(cleaned_docs),
        len(chunks),
        index.ntotal,
    )

    return index, metadata, chunk_embeddings, questions


# ============================================================
# RESULT AGGREGATION
# ============================================================

def compute_k_distribution(summary) -> dict:
    """Count how often each K value was used."""
    counts = Counter(
        r.k_used
        for r in summary.per_question
    )

    return dict(sorted(counts.items()))


def compute_expansion_frequency(summary) -> dict:
    """
    Fraction of questions where the adaptive loop expanded at least once.
    Expansion is identified by num_iterations > 1.
    """

    n = summary.num_questions

    expanded = sum(
        1
        for r in summary.per_question
        if r.num_iterations > 1
    )

    return {
        "expanded_count": expanded,
        "expanded_fraction": round(
            expanded / n,
            4,
        ) if n else 0.0,
        "total_questions": n,
    }


def summary_to_dict(summary) -> dict:
    return {
        "num_questions": summary.num_questions,
        "avg_em": round(summary.avg_em, 4),
        "avg_f1": round(summary.avg_f1, 4),
        "avg_supporting_fact_recall": round(
            summary.avg_supporting_fact_recall,
            4,
        ),
        "avg_k": round(summary.avg_k, 4),
        "avg_iterations": round(
            summary.avg_iterations,
            4,
        ),
        "avg_latency_seconds": round(
            summary.avg_latency_seconds,
            4,
        ),
        "generation_success_rate": round(
            summary.generation_success_rate,
            4,
        ),
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
            "supporting_fact_recall": round(
                r.supporting_fact_recall,
                4,
            ),
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


# ============================================================
# VISUALIZATION 1
# ADAPTIVE K DISTRIBUTION
# ============================================================

def plot_adaptive_k_distribution(k_distribution):
    """
    Shows how often the adaptive system selected each retrieval depth K.
    """

    ks = list(k_distribution.keys())
    counts = list(k_distribution.values())

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        [str(k) for k in ks],
        counts,
    )

    plt.title("Adaptive Retrieval Depth Distribution")
    plt.xlabel("Retrieval depth (K)")
    plt.ylabel("Number of questions")

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    # Display values above bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


# ============================================================
# VISUALIZATION 2
# ANSWER QUALITY COMPARISON
# ============================================================

def plot_quality_comparison(
    adaptive_summary,
    baseline_summary,
):
    """
    Compares Adaptive RAG and Fixed Top-5 on:
      - Exact Match
      - F1
      - Supporting Fact Recall
    """

    metrics = [
        "Exact Match",
        "F1 Score",
        "Supporting Fact Recall",
    ]

    adaptive_values = [
        adaptive_summary.avg_em,
        adaptive_summary.avg_f1,
        adaptive_summary.avg_supporting_fact_recall,
    ]

    baseline_values = [
        baseline_summary.avg_em,
        baseline_summary.avg_f1,
        baseline_summary.avg_supporting_fact_recall,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5))

    bars1 = plt.bar(
        x - width / 2,
        adaptive_values,
        width,
        label="Adaptive RAG",
    )

    bars2 = plt.bar(
        x + width / 2,
        baseline_values,
        width,
        label="Fixed Top-5",
    )

    plt.title("Answer Quality Comparison")
    plt.xlabel("Evaluation metric")
    plt.ylabel("Score")

    plt.xticks(
        x,
        metrics,
    )

    plt.ylim(
        0,
        1,
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


# ============================================================
# VISUALIZATION 3
# RETRIEVAL / EXECUTION EFFICIENCY
# ============================================================

def plot_efficiency_comparison(
    adaptive_summary,
    baseline_summary,
):
    """
    Compares:
      - Average K
      - Average iterations
      - Average latency

    These metrics have different units, so they are shown as
    three separate panels within one figure.
    """

    metric_names = [
        "Average K",
        "Average iterations",
        "Average latency (s)",
    ]

    adaptive_values = [
        adaptive_summary.avg_k,
        adaptive_summary.avg_iterations,
        adaptive_summary.avg_latency_seconds,
    ]

    baseline_values = [
        baseline_summary.avg_k,
        baseline_summary.avg_iterations,
        baseline_summary.avg_latency_seconds,
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4.5),
    )

    for i, ax in enumerate(axes):

        bars = ax.bar(
            ["Adaptive", "Fixed Top-5"],
            [
                adaptive_values[i],
                baseline_values[i],
            ],
        )

        ax.set_title(
            metric_names[i]
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        "Retrieval and Execution Efficiency Comparison"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# VISUALIZATION 4
# EXPANSION FREQUENCY
# ============================================================

def plot_expansion_frequency(expansion_stats):
    """
    Shows the proportion of questions for which the adaptive
    controller expanded retrieval beyond the initial K.
    """

    expanded = expansion_stats["expanded_count"]
    total = expansion_stats["total_questions"]
    not_expanded = total - expanded

    values = [
        expanded,
        not_expanded,
    ]

    labels = [
        "Expanded",
        "No expansion",
    ]

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        labels,
        values,
    )

    plt.title(
        "Adaptive Retrieval Expansion Frequency"
    )

    plt.xlabel(
        "Adaptive retrieval behavior"
    )

    plt.ylabel(
        "Number of questions"
    )

    plt.ylim(
        0,
        max(values) * 1.2 if values else 1,
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(bars, values):
        percentage = (
            value / total * 100
            if total
            else 0
        )

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value} ({percentage:.1f}%)",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


# ============================================================
# GENERATE ALL VISUALIZATIONS
# ============================================================

def generate_visualizations(
    adaptive_summary,
    baseline_summary,
    k_dist_adaptive,
    expansion_stats,
):
    """
    Generate all Review 2 visualizations from the already
    calculated evaluation statistics.

    This function does not modify the evaluation results.
    """

    print("\n" + "=" * 70)
    print("GENERATING EVALUATION VISUALIZATIONS")
    print("=" * 70)

    print("\n1/4 — Adaptive K distribution")
    plot_adaptive_k_distribution(
        k_dist_adaptive
    )

    print("\n2/4 — Answer quality comparison")
    plot_quality_comparison(
        adaptive_summary,
        baseline_summary,
    )

    print("\n3/4 — Efficiency comparison")
    plot_efficiency_comparison(
        adaptive_summary,
        baseline_summary,
    )

    print("\n4/4 — Expansion frequency")
    plot_expansion_frequency(
        expansion_stats
    )

    print("\nAll visualizations generated.")


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    # --------------------------------------------------------
    # Build corpus/index
    # --------------------------------------------------------

    index, metadata, chunk_embeddings, questions = (
        build_pilot_index()
    )

    # --------------------------------------------------------
    # Initialize pipelines
    # --------------------------------------------------------

    adaptive_pipeline = AdaptiveRAGPipeline(
        index,
        metadata,
        chunk_embeddings,
        config_path=str(config_path),
    )

    baseline_pipeline = FixedTopKPipeline(
        index,
        metadata,
        baseline_k=5,
        config_path=str(config_path),
    )

    # --------------------------------------------------------
    # Run Adaptive RAG
    # --------------------------------------------------------

    logger.info(
        "=== Running adaptive pipeline on %d questions ===",
        len(questions),
    )

    adaptive_summary = evaluate_adaptive_pipeline(
        adaptive_pipeline,
        questions,
    )

    # --------------------------------------------------------
    # Run baseline
    # --------------------------------------------------------

    logger.info(
        "=== Running fixed Top-5 baseline on %d questions ===",
        len(questions),
    )

    baseline_summary = evaluate_baseline_pipeline(
        baseline_pipeline,
        questions,
    )

    total_runtime = time.time() - start

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    k_dist_adaptive = compute_k_distribution(
        adaptive_summary
    )

    k_dist_baseline = compute_k_distribution(
        baseline_summary
    )

    expansion_stats = compute_expansion_frequency(
        adaptive_summary
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"STEP 14 PILOT RESULTS — "
        f"{len(questions)} questions, "
        f"HotpotQA distractor/validation"
    )

    print("=" * 70)

    print_comparison(
        adaptive_summary,
        baseline_summary,
    )

    # --------------------------------------------------------
    # K distribution
    # --------------------------------------------------------

    print("\n=== K Distribution (Adaptive) ===")

    for k, count in k_dist_adaptive.items():

        pct = (
            100 * count /
            adaptive_summary.num_questions
        )

        print(
            f"  K={k}: {count} questions "
            f"({pct:.1f}%)"
        )

    print(
        "\n=== K Distribution "
        "(Baseline, always 5 by construction) ==="
    )

    for k, count in k_dist_baseline.items():

        print(
            f"  K={k}: {count} questions"
        )

    # --------------------------------------------------------
    # Expansion frequency
    # --------------------------------------------------------

    print("\n=== Expansion Frequency (Adaptive) ===")

    print(
        "  Questions where the loop expanded "
        "beyond initial K: "
        f"{expansion_stats['expanded_count']}/"
        f"{expansion_stats['total_questions']} "
        f"({expansion_stats['expanded_fraction']:.1%})"
    )

    print(
        f"\nTotal pilot runtime: "
        f"{total_runtime:.1f}s "
        f"({total_runtime / len(questions):.2f}s/question "
        f"average, both pipelines combined)"
    )

    # --------------------------------------------------------
    # Per-question table
    # --------------------------------------------------------

    print("\n=== Per-Question Results (Adaptive) ===")

    print(
        f"{'qid':<26}"
        f"{'EM':<5}"
        f"{'F1':<7}"
        f"{'SF-recall':<11}"
        f"{'K':<4}"
        f"{'iters':<7}"
        f"{'latency(s)':<10}"
    )

    for r in adaptive_summary.per_question:

        print(
            f"{r.question_id:<26}"
            f"{r.em:<5.0f}"
            f"{r.f1:<7.2f}"
            f"{r.supporting_fact_recall:<11.2f}"
            f"{r.k_used:<4}"
            f"{r.num_iterations:<7}"
            f"{r.latency_seconds:<10.2f}"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    results_path = (
        RESULTS_DIR /
        f"pilot_{SAMPLE_SIZE}_{timestamp}.json"
    )

    output = {
        "run_type": f"pilot_{SAMPLE_SIZE}_questions",
        "sample_size": len(questions),

        "notes": (
            "No threshold tuning or logic changes "
            "applied. Pilot run only."
        ),

        "adaptive_summary":
            summary_to_dict(
                adaptive_summary
            ),

        "baseline_summary":
            summary_to_dict(
                baseline_summary
            ),

        "k_distribution_adaptive":
            k_dist_adaptive,

        "k_distribution_baseline":
            k_dist_baseline,

        "expansion_frequency":
            expansion_stats,

        "total_runtime_seconds":
            round(total_runtime, 2),

        "adaptive_per_question":
            per_question_to_dicts(
                adaptive_summary
            ),

        "baseline_per_question":
            per_question_to_dicts(
                baseline_summary
            ),
    }

    with open(
        results_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nFull results saved to: "
        f"{results_path}"
    )

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    generate_visualizations(
        adaptive_summary,
        baseline_summary,
        k_dist_adaptive,
        expansion_stats,
    )

    print(
        "\nOK: Step 14 pilot run complete."
    )


if __name__ == "__main__":
    main()