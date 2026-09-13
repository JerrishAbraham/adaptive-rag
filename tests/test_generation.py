"""
Stage 4 — validation tests for response generation.

Test 1: API connection works (minimal call, no RAG context).
Test 2: A controlled, hand-written context produces a grounded answer.
Test 3: Real optimized context from the adaptive pipeline (Modules 1-3)
        is passed to the generator and produces an answer.

Per the stage spec, this does NOT run full HotpotQA evaluation yet.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import build_generator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_1_api_connection():
    print("\n=== Test 1: API connection ===")
    generator = build_generator()
    result = generator.generate(
        query="Say the single word: OK",
        context=[{"title": "n/a", "text": "No context needed for this connectivity check."}],
    )
    print(f"success={result.success}")
    print(f"answer={result.answer!r}")
    print(f"model={result.model} provider={result.provider} latency={result.latency_seconds}s")
    if not result.success:
        print(f"error={result.error}")
    assert result.success, f"API connection failed: {result.error}"
    assert result.answer != ""
    print("PASS: Test 1")


def test_2_controlled_context():
    print("\n=== Test 2: Controlled context grounding ===")
    generator = build_generator()
    context = [
        {"title": "Scott Derrickson", "text": "Scott Derrickson is an American filmmaker."}
    ]
    result = generator.generate(
        query="What nationality is Scott Derrickson?", context=context
    )
    print(f"success={result.success}")
    print(f"answer={result.answer!r}")
    print(f"input_tokens={result.input_tokens} output_tokens={result.output_tokens}")

    assert result.success
    assert "american" in result.answer.lower(), (
        f"Expected grounded answer to mention 'American', got: {result.answer!r}"
    )
    print("PASS: Test 2")


def test_3_real_pipeline_context():
    print("\n=== Test 3: Real adaptive pipeline context ===")

    from src.data_loader import load_hotpotqa_sample
    from src.preprocessing import preprocess_documents
    from src.chunking import chunk_documents, load_chunking_config
    from src.embeddings import (
        embed_chunks, embeddings_to_matrix, load_embedding_config, load_embedding_model,
    )
    from src.vector_store import build_faiss_index, save_index, load_index
    from src.query_analyzer import analyze_query, load_query_analysis_config
    from src.adaptive_controller import run_initial_retrieval, load_adaptive_controller_config
    from src.context_quality import run_adaptive_loop, load_context_quality_config
    from src.diversity_filter import apply_diversity_filter, load_diversity_config
    import yaml

    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    documents, questions = load_hotpotqa_sample(split="validation", sample_limit=10, config="distractor")
    cleaned_docs = preprocess_documents(documents)
    chunking_cfg = load_chunking_config(str(config_path))
    chunks = chunk_documents(cleaned_docs, chunk_size=chunking_cfg["chunk_size"], chunk_overlap=chunking_cfg["chunk_overlap"])

    embedding_cfg = load_embedding_config(str(config_path))
    embeddings = embed_chunks(chunks, model_name=embedding_cfg["model_name"], normalize=embedding_cfg["normalize_embeddings"])
    matrix = embeddings_to_matrix(chunks, embeddings)
    index = build_faiss_index(matrix)
    index_dir = PROJECT_ROOT / "indexes" / "test_index_module4"
    save_index(index, chunks, index_dir)
    index, metadata = load_index(index_dir)

    qa_config = load_query_analysis_config(str(config_path))
    ac_config = load_adaptive_controller_config(str(config_path))
    quality_config = load_context_quality_config(str(config_path))
    diversity_config = load_diversity_config(str(config_path))

    with open(config_path) as f:
        full_config = yaml.safe_load(f)
    k_values = full_config["retrieval"]["adaptive_k_values"]
    max_k = full_config["retrieval"]["max_k"]
    max_iterations = full_config["retrieval"]["max_iterations"]

    model = load_embedding_model(embedding_cfg["model_name"])
    question = questions[0]

    profile = analyze_query(question.question, qa_config, embedding_model=model, normalize_embedding=embedding_cfg["normalize_embeddings"])
    initial_results, initial_decision = run_initial_retrieval(profile, index, metadata, ac_config, k_values)
    final_results, quality_history, decision_history = run_adaptive_loop(
        profile, index, metadata, initial_results, initial_decision, quality_config, k_values, max_k, max_iterations
    )
    candidate_embeddings = {c.chunk_id: embeddings[c.chunk_id] for c in chunks}
    filtered_results, filter_stats = apply_diversity_filter(
        final_results, candidate_embeddings, profile.embedding,
        lambda_param=diversity_config["lambda"], target_size=diversity_config["target_context_size"],
    )

    print(f"Question: {question.question}")
    print(f"Gold answer: {question.answer}")
    print(f"Optimized context: {len(filtered_results)} chunks -> "
          f"{[r['title'] for r in filtered_results]}")

    generator = build_generator()
    result = generator.generate(query=question.question, context=filtered_results)

    print(f"\nsuccess={result.success}")
    print(f"answer={result.answer!r}")
    print(f"model={result.model} latency={result.latency_seconds}s "
          f"in_tok={result.input_tokens} out_tok={result.output_tokens}")

    assert result.success, f"Generation failed on real pipeline context: {result.error}"
    assert result.answer != ""
    print("PASS: Test 3")


if __name__ == "__main__":
    test_1_api_connection()
    test_2_controlled_context()
    test_3_real_pipeline_context()