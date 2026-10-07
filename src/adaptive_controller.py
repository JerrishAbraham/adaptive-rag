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
    action: str
    reason: str
    iteration: int


def select_initial_k(profile: QueryProfile, config: dict, k_values: list[int]) -> AdaptiveDecision:
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
    decision = select_initial_k(profile, config, k_values)
    results = retrieve_with_k(index, metadata, profile, decision.current_k)
    return results, decision


def load_adaptive_controller_config(config_path: str = "configs/config.yaml") -> dict:
    import yaml

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    return full_config["adaptive_controller"]