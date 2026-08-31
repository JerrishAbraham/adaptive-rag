"""
Module 2, Step 9 — validation test for query analysis.

Runs a set of HotpotQA-style queries with varying expected complexity
through the analyzer and prints the resulting profiles for manual
inspection. Also runs real HotpotQA questions from Module 1's loader
to sanity-check on genuine data.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.query_analyzer import analyze_query, load_query_analysis_config  # noqa: E402
from src.embeddings import load_embedding_model, load_embedding_config  # noqa: E402
from src.data_loader import load_hotpotqa_sample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HANDCRAFTED_QUERIES = [
    "Who wrote Hamlet?",
    "What is the capital of France?",
    "Compare the populations of Tokyo and New York City.",
    "What is the difference between machine learning and deep learning?",
    "Who was the director of the film that starred the actress who played "
    "the lead role in a 1990s television series set in New York?",
    "Why do leaves change color in autumn?",
    "Summarize the causes of World War I.",
]


def test_handcrafted_queries():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    qa_config = load_query_analysis_config(str(config_path))

    print("\n=== Handcrafted queries (no embedding, fast scoring-only check) ===")
    profiles = []
    for q in HANDCRAFTED_QUERIES:
        profile = analyze_query(q, qa_config, embedding_model=None)
        profiles.append(profile)
        print(f"\nQuery: {q}")
        print(f"  intent: {profile.intent}")
        print(f"  complexity_score: {profile.complexity_score}")
        print(f"  features: {profile.features}")

    # Sanity checks on relative ordering, not exact values.
    simple = next(p for p in profiles if p.original_query.startswith("Who wrote"))
    multihop = next(p for p in profiles if "director of the film" in p.original_query)
    comparison = next(p for p in profiles if p.original_query.startswith("Compare"))

    assert multihop.complexity_score > simple.complexity_score, (
        "Multi-hop query should score more complex than a simple factual query."
    )
    assert comparison.intent == "comparison"
    assert multihop.intent == "reasoning_multi_hop"
    assert simple.intent == "factual"

    print("\nOK: handcrafted query scoring behaves as expected.")


def test_real_hotpotqa_questions_with_embeddings():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    qa_config = load_query_analysis_config(str(config_path))
    embedding_cfg = load_embedding_config(str(config_path))

    model = load_embedding_model(embedding_cfg["model_name"])

    documents, questions = load_hotpotqa_sample(
        split="validation", sample_limit=8, config="distractor"
    )

    print("\n=== Real HotpotQA questions (with embeddings) ===")
    for q in questions[:5]:
        profile = analyze_query(
            q.question,
            qa_config,
            embedding_model=model,
            normalize_embedding=embedding_cfg["normalize_embeddings"],
        )
        assert profile.embedding is not None
        assert profile.embedding.shape == (768,)

        print(f"\nQuestion: {q.question}")
        print(f"  hotpotqa type/level: {q.type}/{q.level}")
        print(f"  predicted intent: {profile.intent}")
        print(f"  complexity_score: {profile.complexity_score}")
        print(f"  embedding shape: {profile.embedding.shape}")

    print("\nOK: real HotpotQA questions produce valid embedded profiles.")


if __name__ == "__main__":
    test_handcrafted_queries()
    test_real_hotpotqa_questions_with_embeddings()