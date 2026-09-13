"""
Module 5, Part A — validation test for the end-to-end pipeline.

Builds a small index once, constructs the pipeline, and runs a handful
of real HotpotQA questions through it end-to-end, inspecting the full
PipelineResult for each.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_hotpotqa_sample  # noqa: E402
from src.preprocessing import preprocess_documents  # noqa: E402
from src.chunking import chunk_documents, load_chunking_config  # noqa: E402
from src.embeddings import embed_chunks, embeddings_to_matrix, load_embedding_config  # noqa: E402
from src.vector_store import build_faiss_index, save_index, load_index  # noqa: E402
from src.pipeline import AdaptiveRAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TEST_INDEX_DIR = PROJECT_ROOT / "indexes" / "test_index_pipeline"


def test_end_to_end_pipeline():
    documents, questions = load_hotpotqa_sample(split="validation", sample_limit=15, config="distractor")
    cleaned_docs = preprocess_documents(documents)

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    chunking_cfg = load_chunking_config(str(config_path))
    chunks = chunk_documents(cleaned_docs, chunk_size=chunking_cfg["chunk_size"], chunk_overlap=chunking_cfg["chunk_overlap"])

    embedding_cfg = load_embedding_config(str(config_path))
    embeddings = embed_chunks(chunks, model_name=embedding_cfg["model_name"], normalize=embedding_cfg["normalize_embeddings"])
    matrix = embeddings_to_matrix(chunks, embeddings)
    index = build_faiss_index(matrix)
    save_index(index, chunks, TEST_INDEX_DIR)
    index, metadata = load_index(TEST_INDEX_DIR)

    chunk_embeddings = {c.chunk_id: embeddings[c.chunk_id] for c in chunks}

    pipeline = AdaptiveRAGPipeline(index, metadata, chunk_embeddings, config_path=str(config_path))

    print(f"\nRunning {min(4, len(questions))} questions end-to-end...\n")
    for q in questions[:4]:
        result = pipeline.answer(q.question)

        assert result.generation_success, f"Generation failed: {result.generation_error}"
        assert result.answer != ""
        assert result.final_k in [2, 4, 6, 8]
        assert 0.0 <= result.confidence_score <= 1.0
        assert len(result.context_chunks) > 0

        print(f"Q: {q.question}")
        print(f"  gold answer: {q.answer}")
        print(f"  generated:   {result.answer}")
        print(f"  intent={result.intent} complexity={result.complexity_score} "
              f"k={result.initial_k}->{result.final_k} iterations={result.num_iterations} "
              f"confidence={result.confidence_score}")
        print(f"  context titles: {[c['title'] for c in result.context_chunks]}")
        print(f"  total_latency={result.total_latency_seconds}s "
              f"(generation={result.metadata['generation_latency']}s)")
        print()

    print("OK: end-to-end pipeline test passed.")


if __name__ == "__main__":
    test_end_to_end_pipeline()