"""
Module 3, Steps 11-12 — Context Quality Assessment and Bounded Adaptive Loop.

Computes a Retrieval Confidence Score from retrieval-side signals only
(no LLM call), decides accept vs expand, and runs the bounded
retrieve -> assess -> expand loop with a hard max_iterations and max_k
ceiling. This module owns the loop; Module 2's adaptive_controller
performs only the single initial retrieval that seeds it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.adaptive_controller import AdaptiveDecision, retrieve_with_k
from src.query_analyzer import QueryProfile

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    confidence_score: float
    sufficient: bool
    metrics: dict = field(default_factory=dict)
    reason: str = ""


def _clip_and_scale(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    clipped = max(0.0, min(value, max_value))
    return clipped / max_value


def compute_quality_metrics(results: list[dict]) -> dict:
    """
    Compute raw retrieval-side signals from a list of RetrievalResult
    dicts (as returned by vector_store.search).

    Signals: top similarity, mean similarity, score gap (top1 - top2,
    0 if fewer than 2 results), and unique source document count.
    """
    if not results:
        return {
            "top_similarity": 0.0,
            "mean_similarity": 0.0,
            "score_gap": 0.0,
            "unique_sources": 0,
            "num_results": 0,
        }

    scores = [r["similarity_score"] for r in results]
    top_similarity = scores[0]
    mean_similarity = sum(scores) / len(scores)
    score_gap = (scores[0] - scores[1]) if len(scores) > 1 else 0.0
    unique_sources = len({r["document_id"] for r in results})

    return {
        "top_similarity": top_similarity,
        "mean_similarity": mean_similarity,
        "score_gap": score_gap,
        "unique_sources": unique_sources,
        "num_results": len(results),
    }


def compute_confidence_score(metrics: dict, config: dict) -> float:
    """
    Combine normalized retrieval-side signals into a single 0-1
    Retrieval Confidence Score using configurable weights.

    top_similarity and mean_similarity are already 0-1 (cosine sim on
    normalized embeddings), so they're used directly. score_gap and
    unique_sources are clipped/scaled against configured caps.
    """
    norm_cfg = config["normalization"]
    weights = config["weights"]

    norm_gap = _clip_and_scale(metrics["score_gap"], norm_cfg["max_score_gap"])
    norm_sources = _clip_and_scale(metrics["unique_sources"], norm_cfg["max_unique_sources"])

    score = (
        weights["top_similarity"] * max(0.0, metrics["top_similarity"])
        + weights["mean_similarity"] * max(0.0, metrics["mean_similarity"])
        + weights["score_gap"] * norm_gap
        + weights["unique_sources"] * norm_sources
    )
    return round(min(max(score, 0.0), 1.0), 4)


def assess_quality(results: list[dict], config: dict) -> QualityResult:
    """
    Full quality assessment entry point: compute metrics, combine into a
    confidence score, and decide sufficient vs insufficient against the
    configured threshold.
    """
    metrics = compute_quality_metrics(results)
    confidence = compute_confidence_score(metrics, config)
    threshold = config["confidence_threshold"]
    sufficient = confidence >= threshold

    reason = (
        f"confidence={confidence} {'>=' if sufficient else '<'} threshold={threshold} "
        f"(top_sim={metrics['top_similarity']:.3f}, mean_sim={metrics['mean_similarity']:.3f}, "
        f"gap={metrics['score_gap']:.3f}, unique_sources={metrics['unique_sources']})"
    )

    return QualityResult(
        confidence_score=confidence, sufficient=sufficient, metrics=metrics, reason=reason
    )


def run_adaptive_loop(
    profile: QueryProfile,
    index,
    metadata: list[dict],
    initial_results: list[dict],
    initial_decision: AdaptiveDecision,
    quality_config: dict,
    k_values: list[int],
    max_k: int,
    max_iterations: int,
) -> tuple[list[dict], list[QualityResult], list[AdaptiveDecision]]:
    """
    Bounded retrieve -> assess -> expand loop.

    Starts from the results/decision already produced by Module 2's
    single initial retrieval. If evidence is insufficient and K has not
    reached max_k and iteration count has not reached max_iterations,
    expands K to the next value in k_values and retrieves again.

    Returns:
        (final_results, quality_history, decision_history)
        - quality_history: one QualityResult per iteration assessed
        - decision_history: one AdaptiveDecision per iteration, including
          the original initial_decision as the first entry

    The loop ALWAYS terminates: either sufficient evidence is found, K
    reaches max_k, or max_iterations is reached. No infinite loop path exists.
    """
    results = initial_results
    decision_history = [initial_decision]
    quality_history: list[QualityResult] = []

    current_k = initial_decision.current_k
    iteration = 0

    while True:
        quality = assess_quality(results, quality_config)
        quality_history.append(quality)
        logger.info("Iteration %d assessment: %s", iteration, quality.reason)

        if quality.sufficient:
            decision_history.append(
                AdaptiveDecision(
                    current_k=current_k, next_k=None, action="accept",
                    reason=f"Accepted at iteration {iteration}: {quality.reason}",
                    iteration=iteration,
                )
            )
            break

        if current_k >= max_k:
            decision_history.append(
                AdaptiveDecision(
                    current_k=current_k, next_k=None, action="stop_max_k",
                    reason=f"Stopped: K already at max_k={max_k}. {quality.reason}",
                    iteration=iteration,
                )
            )
            logger.info("Adaptive loop stopped: max_k=%d reached without sufficient confidence.", max_k)
            break

        if iteration + 1 >= max_iterations:
            decision_history.append(
                AdaptiveDecision(
                    current_k=current_k, next_k=None, action="stop_max_iterations",
                    reason=f"Stopped: max_iterations={max_iterations} reached. {quality.reason}",
                    iteration=iteration,
                )
            )
            logger.info("Adaptive loop stopped: max_iterations=%d reached.", max_iterations)
            break

        # Expand: move to the next K value in the bounded set.
        idx = k_values.index(current_k) if current_k in k_values else -1
        next_k = k_values[min(idx + 1, len(k_values) - 1)] if idx >= 0 else min(
            [v for v in k_values if v > current_k], default=max_k
        )
        next_k = min(next_k, max_k)

        decision_history.append(
            AdaptiveDecision(
                current_k=current_k, next_k=next_k, action="expand",
                reason=f"Expanding K {current_k} -> {next_k}: {quality.reason}",
                iteration=iteration,
            )
        )
        logger.info("Expanding retrieval: K %d -> %d", current_k, next_k)

        current_k = next_k
        results = retrieve_with_k(index, metadata, profile, current_k)
        iteration += 1

    return results, quality_history, decision_history


def load_context_quality_config(config_path: str = "configs/config.yaml") -> dict:
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["context_quality"]