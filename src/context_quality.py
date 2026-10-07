from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from src.adaptive_controller import AdaptiveDecision, retrieve_with_k
from src.query_analyzer import QueryProfile

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    confidence_score: float
    sufficient: bool
    metrics: dict = field(default_factory=dict)
    reason: str = ""


def compute_rank_weighted_relevance(scores: list[float]) -> float:

    if not scores:
        return 0.0
    discounts = [1.0 / math.log2(i + 2) for i in range(len(scores))]
    weighted_sum = sum(s * d for s, d in zip(scores, discounts))
    total_weight = sum(discounts)
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def compute_evidence_diversity(embedding_matrix: np.ndarray) -> float:
    
    n = embedding_matrix.shape[0]
    if n < 2:
        return 0.0
    sim_matrix = embedding_matrix @ embedding_matrix.T
    iu = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[iu]
    if pairwise_sims.size == 0:
        return 0.0
    avg_pairwise = float(np.mean(pairwise_sims))
    return float(np.clip(1.0 - np.clip(avg_pairwise, 0.0, 1.0), 0.0, 1.0))


def compute_tail_strength(scores: list[float]) -> float:
    
    if len(scores) < 2:
        return 0.0
    if scores[0] <= 0:
        return 0.0
    return float(np.clip(scores[-1] / scores[0], 0.0, 1.0))


def compute_gate(rwr: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        logger.warning("Gate ceiling (%.3f) <= floor (%.3f); gate forced to 0.", ceiling, floor)
        return 0.0
    return float(np.clip((rwr - floor) / (ceiling - floor), 0.0, 1.0))


def compute_quality_metrics(results: list[dict], chunk_embeddings: dict, config: dict) -> dict:

    if not results:
        return {
            "top_similarity": 0.0, "mean_similarity": 0.0, "score_gap": 0.0,
            "unique_sources": 0, "num_results": 0,
            "rank_weighted_relevance": 0.0, "evidence_diversity": 0.0,
            "tail_strength": 0.0, "gate_value": 0.0, "expansion_pressure": 0.0,
        }

    scores = [r["similarity_score"] for r in results]

    # --- legacy (retained for comparison only; NOT used in the formula) ---
    top_similarity = scores[0]
    mean_similarity = sum(scores) / len(scores)
    score_gap = (scores[0] - scores[1]) if len(scores) > 1 else 0.0
    unique_sources = len({r["document_id"] for r in results})

    # --- new (formula-driving) ---
    rwr = compute_rank_weighted_relevance(scores)

    chunk_ids = [r["chunk_id"] for r in results]
    if len(chunk_ids) >= 2 and all(cid in chunk_embeddings for cid in chunk_ids):
        matrix = np.vstack([chunk_embeddings[cid] for cid in chunk_ids]).astype(np.float32)
        evidence_diversity = compute_evidence_diversity(matrix)
    else:
        if len(chunk_ids) >= 2:
            logger.warning(
                "compute_quality_metrics: %d/%d chunk embeddings missing; evidence_diversity defaulted to 0.0",
                sum(1 for cid in chunk_ids if cid not in chunk_embeddings), len(chunk_ids),
            )
        evidence_diversity = 0.0

    tail_strength = compute_tail_strength(scores)
    gate_cfg = config.get("gate", {"floor": 0.45, "ceiling": 0.75})
    gate_value = compute_gate(rwr, gate_cfg["floor"], gate_cfg["ceiling"])
    expansion_pressure = gate_value * tail_strength

    return {
        "top_similarity": round(top_similarity, 4),
        "mean_similarity": round(mean_similarity, 4),
        "score_gap": round(score_gap, 4),
        "unique_sources": unique_sources,
        "num_results": len(results),
        "rank_weighted_relevance": round(rwr, 4),
        "evidence_diversity": round(evidence_diversity, 4),
        "tail_strength": round(tail_strength, 4),
        "gate_value": round(gate_value, 4),
        "expansion_pressure": round(expansion_pressure, 4),
    }


def compute_confidence_score(metrics: dict, config: dict) -> float:
  
    weights = config["weights"]
    score = (
        weights["rank_weighted_relevance"] * metrics["rank_weighted_relevance"]
        + weights["evidence_diversity"] * metrics["evidence_diversity"]
    )
    return round(min(max(score, 0.0), 1.0), 4)


def assess_quality(results: list[dict], config: dict, chunk_embeddings: dict | None = None) -> QualityResult:
    
    chunk_embeddings = chunk_embeddings or {}
    metrics = compute_quality_metrics(results, chunk_embeddings, config)
    confidence = compute_confidence_score(metrics, config)
    threshold = config["confidence_threshold"]
    sufficient = confidence >= threshold

    reason = (
        f"confidence={confidence} {'>=' if sufficient else '<'} threshold={threshold} "
        f"(RWR={metrics['rank_weighted_relevance']:.3f}, "
        f"ED={metrics['evidence_diversity']:.3f}, "
        f"expansion_pressure={metrics['expansion_pressure']:.3f} "
        f"[gate={metrics['gate_value']:.3f}, tail={metrics['tail_strength']:.3f}]; "
        f"legacy: top_sim={metrics['top_similarity']:.3f}, mean_sim={metrics['mean_similarity']:.3f}, "
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
    chunk_embeddings: dict | None = None,
) -> tuple[list[dict], list[QualityResult], list[AdaptiveDecision]]:
   
    chunk_embeddings = chunk_embeddings or {}
    results = initial_results
    decision_history = [initial_decision]
    quality_history: list[QualityResult] = []

    current_k = initial_decision.current_k
    iteration = 0

    while True:
        quality = assess_quality(results, quality_config, chunk_embeddings)
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

        expansion_cfg = quality_config.get("expansion", {})
        min_pressure = expansion_cfg.get("min_pressure", 0.50)
        expansion_pressure = quality.metrics.get("expansion_pressure", 0.0)

        if expansion_pressure < min_pressure:
            decision_history.append(
                AdaptiveDecision(
                    current_k=current_k, next_k=None,
                    action="stop_low_expansion_pressure",
                    reason=(
                        f"Stopped: expansion_pressure={expansion_pressure:.3f} "
                        f"< min_pressure={min_pressure:.3f}. {quality.reason}"
                    ),
                    iteration=iteration,
                )
            )
            logger.info(
                "Adaptive loop stopped: expansion_pressure=%.3f below min_pressure=%.3f.",
                expansion_pressure,
                min_pressure,
            )
            break

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