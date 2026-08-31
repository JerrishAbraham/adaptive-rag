"""
Module 2, Step 10 — Adaptive Retrieval Controller.

Consumes a QueryProfile (from query_analyzer.py) to choose an initial K
from the configured bounded set, performs the first FAISS retrieval via
vector_store.search, and records an AdaptiveDecision explaining why that
K was chosen. This stage performs exactly one retrieval iteration — the
expand/reassess loop is Module 3 (Context Quality Assessment), which
consumes what this module produces.

The LLM is not involved anywhere in this module, per the locked scope:
retrieval depth is decided by query + retrieval-side signals only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.query_analyzer import QueryProfile
from src.vector_store import search as faiss_search

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveDecision:
    current_k: int
    next_k: int | None
    action: str          # "initial_select" | "expand" | "accept" | "stop_max_k"
    reason: str
    iteration: int


def select_initial_k(profile: QueryProfile, config: dict, k_values: list[int]) -> AdaptiveDecision:
    """
    Choose an initial K using the complexity_score band plus an intent-based
    bump, both fully configurable and logged with an explicit reason.

    Args:
        profile: QueryProfile from query_analyzer.analyze_query.
        config: the 'adaptive_controller' section of config.yaml.
        k_values: the bounded set of allowed K values (retrieval.adaptive_k_values).
    """
    bands = config["complexity_bands"]
    chosen_k = None
    band_reason = None

    for band in bands:
        if profile.complexity_score <= band["max_score"]:
            chosen_k = band["initial_k"]
            band_reason = f"complexity_score={profile.complexity_score} <= {band['max_score']}"
            break

    if chosen_k is None:
        chosen_k = bands[-1]["initial_k"]
        band_reason = f"complexity_score={profile.complexity_score} exceeded all bands; using max band"

    reason_parts = [f"initial K={chosen_k} from complexity band ({band_reason})"]

    bump = config.get("intent_bump", {}).get(profile.intent, 0)
    if bump > 0 and chosen_k in k_values:
        idx = k_values.index(chosen_k)
        new_idx = min(idx + bump, len(k_values) - 1)
        if new_idx != idx:
            bumped_k = k_values[new_idx]
            reason_parts.append(
                f"bumped {chosen_k} -> {bumped_k} due to intent='{profile.intent}' (+{bump} step)"
            )
            chosen_k = bumped_k

    # Safety clamp: always land on an allowed value even if config bands
    # are misconfigured with an off-list K.
    if chosen_k not in k_values:
        closest = min(k_values, key=lambda v: abs(v - chosen_k))
        reason_parts.append(f"clamped {chosen_k} -> {closest} (not in allowed k_values)")
        chosen_k = closest

    decision = AdaptiveDecision(
        current_k=chosen_k,
        next_k=None,
        action="initial_select",
        reason="; ".join(reason_parts),
        iteration=0,
    )
    logger.info("Initial K decision: %s", decision.reason)
    return decision


def retrieve_with_k(index, metadata: list[dict], profile: QueryProfile, k: int) -> list[dict]:
    """
    Perform one FAISS retrieval call using the query embedding already
    attached to the QueryProfile (from Step 9). Requires profile.embedding
    to be set — i.e. analyze_query must have been called with an
    embedding_model, not None.
    """
    if profile.embedding is None:
        raise ValueError(
            "QueryProfile has no embedding. Call analyze_query with an "
            "embedding_model to enable retrieval."
        )
    results = faiss_search(index, metadata, profile.embedding, k=k)
    logger.info(
        "Retrieved %d results for k=%d (query: %r)",
        len(results), k, profile.cleaned_query[:60],
    )
    return results


def run_initial_retrieval(
    profile: QueryProfile,
    index,
    metadata: list[dict],
    config: dict,
    k_values: list[int],
) -> tuple[list[dict], AdaptiveDecision]:
    """
    Full Step 10 entry point: select initial K, then perform the first
    retrieval. Returns (retrieval_results, decision) so both the evidence
    and the explanation for why that evidence was fetched are available
    to later stages (Module 3 quality assessment, and eventually logging).
    """
    decision = select_initial_k(profile, config, k_values)
    results = retrieve_with_k(index, metadata, profile, decision.current_k)
    return results, decision


def load_adaptive_controller_config(config_path: str = "configs/config.yaml") -> dict:
    import yaml

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    return full_config["adaptive_controller"]