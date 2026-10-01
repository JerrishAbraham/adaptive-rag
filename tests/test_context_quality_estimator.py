import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.context_quality import (
    compute_rank_weighted_relevance,
    compute_evidence_diversity,
    compute_tail_strength,
    compute_gate,
    compute_quality_metrics,
    compute_confidence_score,
    assess_quality,
)

CONFIG = {
    "confidence_threshold": 0.55,
    "weights": {
        # Confidence measures context quality only.
        # Expansion pressure is a separate signal for the adaptive controller.
        "rank_weighted_relevance": 0.70,
        "evidence_diversity": 0.30,
    },
    "expansion": {
        "min_pressure": 0.50,
    },
    "gate": {"floor": 0.45, "ceiling": 0.75},
}


def _unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _make_result(chunk_id, doc_id, score):
    return {
        "chunk_id": chunk_id, "document_id": doc_id, "title": doc_id,
        "text": "placeholder", "rank": 0, "similarity_score": score,
    }


def test_rank_weighted_relevance_empty():
    assert compute_rank_weighted_relevance([]) == 0.0


def test_rank_weighted_relevance_single_score_equals_the_score():
    assert abs(compute_rank_weighted_relevance([0.8]) - 0.8) < 1e-9


def test_rank_weighted_relevance_weights_top_ranks_more():
    front_loaded = compute_rank_weighted_relevance([0.9, 0.9, 0.2, 0.2])
    back_loaded = compute_rank_weighted_relevance([0.2, 0.2, 0.9, 0.9])
    assert front_loaded > back_loaded


def test_gate_below_floor_is_zero():
    assert compute_gate(rwr=0.3, floor=0.45, ceiling=0.75) == 0.0


def test_gate_at_or_above_ceiling_is_one():
    assert compute_gate(rwr=0.75, floor=0.45, ceiling=0.75) == 1.0
    assert compute_gate(rwr=0.9, floor=0.45, ceiling=0.75) == 1.0


def test_gate_linear_between_floor_and_ceiling():
    mid = compute_gate(rwr=0.60, floor=0.45, ceiling=0.75)
    assert abs(mid - 0.5) < 1e-9


def test_k1_no_diversity_no_expansion_pressure():
    results = [_make_result("c1", "docA", 0.8)]
    embeddings = {"c1": _unit([1, 0, 0, 0])}

    metrics = compute_quality_metrics(results, embeddings, CONFIG)

    assert metrics["evidence_diversity"] == 0.0
    assert metrics["tail_strength"] == 0.0
    assert metrics["expansion_pressure"] == 0.0

    confidence = compute_confidence_score(metrics, CONFIG)
    expected = round(
        CONFIG["weights"]["rank_weighted_relevance"]
        * metrics["rank_weighted_relevance"],
        4,
    )
    assert abs(confidence - expected) < 1e-6


def test_duplicate_chunks_penalize_diversity():
    results = [
        _make_result("c1", "docA", 0.75),
        _make_result("c2", "docB", 0.74),
        _make_result("c3", "docC", 0.73),
    ]
    same_vec = _unit([1, 0, 0, 0])
    embeddings = {"c1": same_vec, "c2": same_vec.copy(), "c3": same_vec.copy()}

    metrics = compute_quality_metrics(results, embeddings, CONFIG)

    assert metrics["evidence_diversity"] < 0.05
    assert metrics["unique_sources"] == 3

    diverse_embeddings = {
        "c1": _unit([1, 0, 0, 0]),
        "c2": _unit([0, 1, 0, 0]),
        "c3": _unit([0, 0, 1, 0]),
    }
    diverse_metrics = compute_quality_metrics(results, diverse_embeddings, CONFIG)
    assert diverse_metrics["evidence_diversity"] > metrics["evidence_diversity"]


def test_uniformly_weak_results_get_low_confidence_without_expansion_penalty():
    results = [
        _make_result("c1", "docA", 0.30),
        _make_result("c2", "docB", 0.29),
        _make_result("c3", "docC", 0.28),
        _make_result("c4", "docD", 0.27),
    ]
    embeddings = {
        "c1": _unit([1, 0, 0, 0]), "c2": _unit([0, 1, 0, 0]),
        "c3": _unit([0, 0, 1, 0]), "c4": _unit([0, 0, 0, 1]),
    }

    metrics = compute_quality_metrics(results, embeddings, CONFIG)

    assert metrics["rank_weighted_relevance"] < CONFIG["gate"]["floor"]
    assert metrics["gate_value"] == 0.0
    assert metrics["expansion_pressure"] == 0.0

    confidence = compute_confidence_score(metrics, CONFIG)
    result = assess_quality(results, CONFIG, embeddings)
    assert result.sufficient is False
    assert confidence < CONFIG["confidence_threshold"]


def test_strong_top_weak_tail_has_small_expansion_pressure():
    results = [
        _make_result("c1", "docA", 0.85),
        _make_result("c2", "docB", 0.40),
        _make_result("c3", "docC", 0.30),
        _make_result("c4", "docD", 0.25),
    ]
    embeddings = {
        "c1": _unit([1, 0, 0, 0]), "c2": _unit([0.6, 0.8, 0, 0]),
        "c3": _unit([0, 1, 0, 0]), "c4": _unit([0, 0, 1, 0]),
    }

    metrics = compute_quality_metrics(results, embeddings, CONFIG)

    assert metrics["tail_strength"] < 0.35
    assert metrics["expansion_pressure"] < 0.30

    result = assess_quality(results, CONFIG, embeddings)
    print(f"\n[strong-top-weak-tail] confidence={result.confidence_score} metrics={metrics}")


def test_multiple_strong_complementary_results_trigger_expansion_pressure():
    results = [
        _make_result("c1", "docA", 0.80),
        _make_result("c2", "docB", 0.78),
        _make_result("c3", "docC", 0.77),
        _make_result("c4", "docD", 0.76),
    ]
    embeddings = {
        "c1": _unit([1, 0, 0, 0]), "c2": _unit([0, 1, 0, 0]),
        "c3": _unit([0, 0, 1, 0]), "c4": _unit([0, 0, 0, 1]),
    }

    metrics = compute_quality_metrics(results, embeddings, CONFIG)

    assert metrics["rank_weighted_relevance"] >= CONFIG["gate"]["ceiling"] - 0.05
    assert metrics["gate_value"] > 0.8
    assert metrics["tail_strength"] > 0.85
    assert metrics["expansion_pressure"] > 0.5
    assert metrics["evidence_diversity"] > 0.9

    result = assess_quality(results, CONFIG, embeddings)
    print(f"\n[multi-strong-complementary] confidence={result.confidence_score} "
          f"sufficient={result.sufficient} metrics={metrics}")


def test_expansion_pressure_does_not_change_confidence():
    """
    Confidence measures current context quality only. Expansion pressure is
    intentionally excluded from the confidence formula.
    """
    results = [
        _make_result("c1", "docA", 0.80),
        _make_result("c2", "docB", 0.78),
        _make_result("c3", "docC", 0.77),
        _make_result("c4", "docD", 0.76),
    ]
    embeddings = {
        "c1": _unit([1, 0, 0, 0]), "c2": _unit([0, 1, 0, 0]),
        "c3": _unit([0, 0, 1, 0]), "c4": _unit([0, 0, 0, 1]),
    }

    metrics = compute_quality_metrics(results, embeddings, CONFIG)

    low_pressure_metrics = dict(metrics)
    low_pressure_metrics["expansion_pressure"] = 0.0

    high_pressure_metrics = dict(metrics)
    high_pressure_metrics["expansion_pressure"] = 1.0

    low_pressure_confidence = compute_confidence_score(
        low_pressure_metrics, CONFIG
    )
    high_pressure_confidence = compute_confidence_score(
        high_pressure_metrics, CONFIG
    )

    assert low_pressure_confidence == high_pressure_confidence


def test_assess_quality_empty_results():
    result = assess_quality([], CONFIG, {})
    assert result.confidence_score == 0.0
    assert result.sufficient is False
    assert result.metrics["num_results"] == 0


def test_assess_quality_missing_embeddings_fails_soft():
    results = [_make_result("c1", "docA", 0.7), _make_result("c2", "docB", 0.65)]
    result = assess_quality(results, CONFIG, {})
    assert result.metrics["evidence_diversity"] == 0.0
    assert isinstance(result.confidence_score, float)


def test_legacy_metrics_present_but_do_not_affect_confidence():
    results = [
        _make_result("c1", "docA", 0.9),
        _make_result("c2", "docB", 0.1),
    ]
    embeddings = {"c1": _unit([1, 0, 0, 0]), "c2": _unit([0, 1, 0, 0])}

    metrics = compute_quality_metrics(results, embeddings, CONFIG)
    assert "top_similarity" in metrics and "score_gap" in metrics and "unique_sources" in metrics
    assert metrics["score_gap"] == 0.8

    mutated = dict(metrics)
    mutated["score_gap"] = 999.0
    mutated["unique_sources"] = 999
    mutated["top_similarity"] = 0.0
    mutated["mean_similarity"] = 0.0

    original_confidence = compute_confidence_score(metrics, CONFIG)
    mutated_confidence = compute_confidence_score(mutated, CONFIG)
    assert original_confidence == mutated_confidence