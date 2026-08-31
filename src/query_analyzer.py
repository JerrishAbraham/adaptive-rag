"""
Module 2, Step 9 — Query Analysis: Complexity and Intent Scoring.

Builds a QueryProfile for an incoming query using only lightweight,
deterministic, reproducible features (query length, naive entity count,
comparison/multi-hop cue words) — no LLM call, per the locked scope.

This stage does NOT select an initial K or perform retrieval. That is
Module 2, Step 10 (Adaptive Retrieval Controller), built on top of the
QueryProfile this module produces.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

VALID_INTENTS = {"factual", "explanatory", "comparison", "summarization", "reasoning_multi_hop"}


@dataclass
class QueryProfile:
    original_query: str
    cleaned_query: str
    embedding: Optional[np.ndarray]
    complexity_score: float
    intent: str
    features: dict = field(default_factory=dict)


def clean_query(query: str) -> str:
    """Normalize whitespace only; wording/meaning is never altered."""
    text = re.sub(r"\s+", " ", query.strip())
    return text


def _count_naive_entities(query: str) -> int:
    """
    Heuristic entity/concept count: capitalized words not at the start
    of the sentence, treated as a proxy for named entities/concepts.

    Not true NER — deliberately lightweight per the reference doc's
    requirement to avoid an additional model/LLM call for query analysis.
    Multi-word proper nouns (e.g. "New York City") are counted as one
    entity by collapsing consecutive capitalized words.
    """
    words = query.split()
    if not words:
        return 0

    count = 0
    in_entity_run = False
    for i, word in enumerate(words):
        stripped = word.strip(".,?!;:\"'()")
        is_cap = stripped[:1].isupper() and stripped[:1].isalpha()
        # Skip the very first word — sentence-initial capitalization is
        # not a reliable entity signal ("What was...", "Who directed...").
        if i == 0:
            in_entity_run = False
            continue
        if is_cap:
            if not in_entity_run:
                count += 1
            in_entity_run = True
        else:
            in_entity_run = False
    return count


def _count_cue_words(query_lower: str, cue_words: list[str]) -> int:
    return sum(1 for cue in cue_words if cue.lower() in query_lower)


def _clip_and_scale(value: float, max_value: float) -> float:
    """Clip to [0, max_value] then scale to [0, 1]."""
    if max_value <= 0:
        return 0.0
    clipped = max(0.0, min(value, max_value))
    return clipped / max_value


def extract_features(query: str, config: dict) -> dict:
    """
    Extract deterministic complexity/intent features from a cleaned query.

    Returns a dict of raw and normalized feature values, so the scoring
    weights and thresholds remain visible/auditable rather than hidden
    inside a single opaque score.
    """
    query_lower = query.lower()
    word_count = len(query.split())
    entity_count = _count_naive_entities(query)

    norm_cfg = config["normalization"]
    comparison_hits = _count_cue_words(query_lower, config["comparison_cue_words"])
    multi_hop_hits = _count_cue_words(query_lower, config["multi_hop_cue_words"])

    return {
        "word_count": word_count,
        "entity_count": entity_count,
        "comparison_cue_hits": comparison_hits,
        "multi_hop_cue_hits": multi_hop_hits,
        "norm_word_count": _clip_and_scale(word_count, norm_cfg["max_word_count"]),
        "norm_entity_count": _clip_and_scale(entity_count, norm_cfg["max_entity_count"]),
        "norm_comparison": 1.0 if comparison_hits > 0 else 0.0,
        "norm_multi_hop": 1.0 if multi_hop_hits >= 2 else (0.5 if multi_hop_hits == 1 else 0.0),
    }


def compute_complexity_score(features: dict, weights: dict) -> float:
    """
    Weighted combination of normalized features into a single 0-1 score.

    Deterministic and fully explainable: given the same weights and the
    same features dict, the score is always reproducible and each term's
    contribution can be inspected directly.
    """
    score = (
        weights["length"] * features["norm_word_count"]
        + weights["entity_count"] * features["norm_entity_count"]
        + weights["multi_hop_cues"] * features["norm_multi_hop"]
        + weights["comparison_cues"] * features["norm_comparison"]
    )
    return round(min(max(score, 0.0), 1.0), 4)


def classify_intent(query: str, features: dict, config: dict) -> str:
    """
    Rule-based intent classification. Order matters: more specific
    categories are checked before falling back to 'factual'.
    """
    query_lower = query.lower()
    intent_keywords = config.get("intent_keywords", {})

    if features["comparison_cue_hits"] > 0:
        return "comparison"

    if features["multi_hop_cue_hits"] >= 2 and features["word_count"] >= 15:
        return "reasoning_multi_hop"

    for keyword in intent_keywords.get("summarization", []):
        if keyword in query_lower:
            return "summarization"

    for keyword in intent_keywords.get("explanatory", []):
        if keyword in query_lower:
            return "explanatory"

    return "factual"


def analyze_query(
    query: str,
    config: dict,
    embedding_model=None,
    normalize_embedding: bool = True,
) -> QueryProfile:
    """
    Full Step 9 entry point: clean the query, extract features, score
    complexity, classify intent, and optionally attach a query embedding.

    Args:
        query: the raw incoming query string.
        config: the 'query_analysis' section of config.yaml.
        embedding_model: an already-loaded SentenceTransformer (from
            embeddings.load_embedding_model), or None to skip embedding
            (useful for fast unit tests of the scoring logic alone).
        normalize_embedding: passed through to model.encode.
    """
    original_query = query
    cleaned = clean_query(query)

    features = extract_features(cleaned, config)
    complexity_score = compute_complexity_score(features, config["weights"])
    intent = classify_intent(cleaned, features, config)
    assert intent in VALID_INTENTS, f"Unexpected intent label: {intent}"

    embedding = None
    if embedding_model is not None:
        embedding = embedding_model.encode(
            [cleaned], normalize_embeddings=normalize_embedding, convert_to_numpy=True
        )[0]

    profile = QueryProfile(
        original_query=original_query,
        cleaned_query=cleaned,
        embedding=embedding,
        complexity_score=complexity_score,
        intent=intent,
        features=features,
    )

    logger.debug(
        "QueryProfile: intent=%s complexity=%.3f features=%s",
        profile.intent, profile.complexity_score, profile.features,
    )
    return profile


def load_query_analysis_config(config_path: str = "configs/config.yaml") -> dict:
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["query_analysis"]