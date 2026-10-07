
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
   
    return vectors @ vectors.T


def mmr_select(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_texts: list[str],
    lambda_param: float,
    target_size: int,
) -> list[int]:
    
    n = candidate_embeddings.shape[0]
    target_size = min(target_size, n)
    if n == 0 or target_size == 0:
        return []

    relevance = candidate_embeddings @ query_embedding  # shape (N,)
    pairwise_sim = _cosine_sim_matrix(candidate_embeddings)  # shape (N, N)

    selected: list[int] = []
    remaining = set(range(n))

    # First pick: highest relevance, no diversity penalty possible yet.
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.discard(first)

    while len(selected) < target_size and remaining:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            max_sim_to_selected = max(pairwise_sim[idx, s] for s in selected)
            mmr_score = lambda_param * relevance[idx] - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def apply_diversity_filter(
    results: list[dict],
    candidate_embeddings: dict,
    query_embedding: np.ndarray,
    lambda_param: float,
    target_size: int,
) -> tuple[list[dict], dict]:
    
    if not results:
        return [], {"pre_filter_count": 0, "post_filter_count": 0, "selected_chunk_ids": []}

    chunk_ids = [r["chunk_id"] for r in results]
    matrix = np.vstack([candidate_embeddings[cid] for cid in chunk_ids]).astype(np.float32)

    selected_local_indices = mmr_select(
        query_embedding=query_embedding.astype(np.float32),
        candidate_embeddings=matrix,
        candidate_texts=[r["text"] for r in results],
        lambda_param=lambda_param,
        target_size=target_size,
    )

    filtered_results = [results[i] for i in selected_local_indices]
    selected_chunk_ids = [chunk_ids[i] for i in selected_local_indices]

    stats = {
        "pre_filter_count": len(results),
        "post_filter_count": len(filtered_results),
        "selected_chunk_ids": selected_chunk_ids,
    }
    logger.info(
        "Diversity filter: %d -> %d chunks (lambda=%.2f, target_size=%d)",
        stats["pre_filter_count"], stats["post_filter_count"], lambda_param, target_size,
    )
    return filtered_results, stats


def load_diversity_config(config_path: str = "configs/config.yaml") -> dict:
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["diversity"]