"""
Module 5 — Fixed Top-5 Baseline Pipeline.

Implements the locked baseline: always retrieves exactly K=5 chunks via
plain FAISS similarity search, no adaptive logic, no quality assessment
loop, no diversity filtering. Used as the comparison point against the
adaptive pipeline, per the project's locked scope (retrieval.baseline_k: 5).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.vector_store import search as faiss_search
from src.generator import Generator, load_generator_config
from src.embeddings import load_embedding_model, load_embedding_config

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    question: str
    answer: str
    k: int
    context_chunks: list[dict]
    generation_success: bool
    generation_error: str | None
    total_latency_seconds: float
    metadata: dict = field(default_factory=dict)


class FixedTopKPipeline:
    """Fixed Top-K RAG baseline: no adaptive retrieval, no expansion, no diversity filtering."""

    def __init__(self, index, metadata: list[dict], baseline_k: int = 5, config_path: str = "configs/config.yaml"):
        self.index = index
        self.metadata = metadata
        self.baseline_k = baseline_k

        embedding_cfg = load_embedding_config(config_path)
        self.embedding_model = load_embedding_model(embedding_cfg["model_name"])
        self.normalize_embeddings = embedding_cfg["normalize_embeddings"]

        generator_config = load_generator_config(config_path)
        self.generator = Generator(generator_config)

        logger.info("FixedTopKPipeline initialized: k=%d", baseline_k)

    def answer(self, question: str) -> BaselineResult:
        start = time.time()

        query_vector = self.embedding_model.encode(
            [question], normalize_embeddings=self.normalize_embeddings, convert_to_numpy=True
        )[0]
        results = faiss_search(self.index, self.metadata, query_vector, k=self.baseline_k)

        gen_result = self.generator.generate(query=question, context=results)
        total_latency = time.time() - start

        result = BaselineResult(
            question=question,
            answer=gen_result.answer,
            k=self.baseline_k,
            context_chunks=results,
            generation_success=gen_result.success,
            generation_error=gen_result.error,
            total_latency_seconds=round(total_latency, 3),
            metadata={
                "generation_latency": gen_result.latency_seconds,
                "input_tokens": gen_result.input_tokens,
                "output_tokens": gen_result.output_tokens,
            },
        )
        return result