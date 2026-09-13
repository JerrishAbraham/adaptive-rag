"""
End-to-end Adaptive RAG pipeline.

Wires together Modules 1-4 into a single callable function per query:

Query -> Query Analysis -> Adaptive Retrieval -> Context Quality
      -> Optional Expansion -> Diversity Filtering -> Prompt Construction
      -> Generator -> Answer

This module performs NO new logic of its own — it only orchestrates
calls into the existing, independently-tested modules. Knowledge base
construction (Module 1's indexing) happens once, outside this function,
and the resulting (index, metadata, chunk_embeddings) are passed in.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import yaml

from src.query_analyzer import analyze_query, load_query_analysis_config
from src.adaptive_controller import run_initial_retrieval, load_adaptive_controller_config
from src.context_quality import run_adaptive_loop, load_context_quality_config
from src.diversity_filter import apply_diversity_filter, load_diversity_config
from src.generator import Generator, load_generator_config
from src.embeddings import load_embedding_model, load_embedding_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    question: str
    answer: str
    intent: str
    complexity_score: float
    initial_k: int
    final_k: int
    num_iterations: int
    confidence_score: float
    context_chunks: list[dict]
    generation_success: bool
    generation_error: str | None
    total_latency_seconds: float
    metadata: dict = field(default_factory=dict)


class AdaptiveRAGPipeline:
    """
    Loads all stage configs once and exposes a single `answer(question)`
    entry point. Meant to be constructed once per process (embedding
    model + generator client are reused across calls) and reused across
    many questions during evaluation (Module 5, Part B).
    """

    def __init__(self, index, metadata: list[dict], chunk_embeddings: dict, config_path: str = "configs/config.yaml"):
        """
        Args:
            index: a loaded FAISS index (from vector_store.load_index).
            metadata: the parallel chunk metadata list.
            chunk_embeddings: dict chunk_id -> np.ndarray, needed by the
                diversity filter stage.
            config_path: path to config.yaml.
        """
        self.index = index
        self.metadata = metadata
        self.chunk_embeddings = chunk_embeddings

        with open(config_path) as f:
            self.full_config = yaml.safe_load(f)

        self.qa_config = load_query_analysis_config(config_path)
        self.ac_config = load_adaptive_controller_config(config_path)
        self.quality_config = load_context_quality_config(config_path)
        self.diversity_config = load_diversity_config(config_path)

        self.k_values = self.full_config["retrieval"]["adaptive_k_values"]
        self.max_k = self.full_config["retrieval"]["max_k"]
        self.max_iterations = self.full_config["retrieval"]["max_iterations"]

        embedding_cfg = load_embedding_config(config_path)
        self.embedding_model_name = embedding_cfg["model_name"]
        self.normalize_embeddings = embedding_cfg["normalize_embeddings"]
        self.embedding_model = load_embedding_model(self.embedding_model_name)

        generator_config = load_generator_config(config_path)
        self.generator = Generator(generator_config)

        logger.info("AdaptiveRAGPipeline initialized: %d chunks in index", index.ntotal)

    def answer(self, question: str) -> PipelineResult:
        """
        Run the full pipeline for a single question and return a
        PipelineResult capturing the answer plus every intermediate
        decision, for later evaluation/analysis (Module 5, Part B).
        """
        start = time.time()

        # Step 1: Query Analysis
        profile = analyze_query(
            question, self.qa_config, embedding_model=self.embedding_model,
            normalize_embedding=self.normalize_embeddings,
        )

        # Step 2: Adaptive Retrieval (initial)
        initial_results, initial_decision = run_initial_retrieval(
            profile, self.index, self.metadata, self.ac_config, self.k_values
        )

        # Step 3: Context Quality Assessment + Optional Expansion
        final_results, quality_history, decision_history = run_adaptive_loop(
            profile, self.index, self.metadata, initial_results, initial_decision,
            self.quality_config, self.k_values, self.max_k, self.max_iterations,
        )

        # Step 4: Diversity Filtering
        filtered_results, filter_stats = apply_diversity_filter(
            final_results, self.chunk_embeddings, profile.embedding,
            lambda_param=self.diversity_config["lambda"],
            target_size=self.diversity_config["target_context_size"],
        )

        # Step 5: Generation (prompt construction happens inside generator.generate)
        gen_result = self.generator.generate(query=question, context=filtered_results)

        total_latency = time.time() - start

        num_iterations = sum(1 for d in decision_history if d.action == "expand") + 1
        final_k = decision_history[-1].current_k

        result = PipelineResult(
            question=question,
            answer=gen_result.answer,
            intent=profile.intent,
            complexity_score=profile.complexity_score,
            initial_k=initial_decision.current_k,
            final_k=final_k,
            num_iterations=num_iterations,
            confidence_score=quality_history[-1].confidence_score,
            context_chunks=filtered_results,
            generation_success=gen_result.success,
            generation_error=gen_result.error,
            total_latency_seconds=round(total_latency, 3),
            metadata={
                "decision_history": [d.reason for d in decision_history],
                "filter_stats": filter_stats,
                "generation_latency": gen_result.latency_seconds,
                "input_tokens": gen_result.input_tokens,
                "output_tokens": gen_result.output_tokens,
            },
        )
        logger.info(
            "Pipeline complete: intent=%s k=%d->%d confidence=%.3f gen_success=%s latency=%.2fs",
            result.intent, result.initial_k, result.final_k, result.confidence_score,
            result.generation_success, result.total_latency_seconds,
        )
        return result