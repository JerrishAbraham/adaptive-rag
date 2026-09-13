"""
Module 5, Part B — Evaluation Metrics and Batch Harness.

Computes:
  - Generation quality: Exact Match (EM) and token-level F1, standard
    SQuAD-style normalization (lowercase, strip punctuation/articles,
    collapse whitespace).
  - Retrieval quality: supporting-fact title recall — fraction of a
    question's gold supporting-fact article titles present among the
    retrieved chunks' titles. Chosen over generic Recall@K because
    HotpotQA's supporting_facts are the ground-truth evidence signal
    the dataset actually provides; there is no separate relevance
    judgment file to compute classic Recall@K against.
  - Efficiency: average K used, average latency, average tokens.

Runs both the adaptive pipeline and the fixed Top-5 baseline over the
same question set and reports a side-by-side comparison.
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Generation metrics (EM / F1) — standard SQuAD-style normalization
# ---------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, remove articles, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match_score(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------
# Retrieval metric — supporting-fact title recall
# ---------------------------------------------------------------------

def supporting_fact_title_recall(context_chunks: list[dict], supporting_facts: list[tuple[str, int]]) -> float:
    """
    Fraction of unique gold supporting-fact article titles that appear
    among the titles of the retrieved/context chunks.

    Returns 1.0 if there are no supporting facts to check (edge case,
    avoids division by zero) and 0.0 if context is empty but gold
    supporting facts exist.
    """
    gold_titles = {t for t, _ in supporting_facts}
    if not gold_titles:
        return 1.0
    if not context_chunks:
        return 0.0

    retrieved_titles = {c["title"] for c in context_chunks}
    hit = gold_titles & retrieved_titles
    return len(hit) / len(gold_titles)


# ---------------------------------------------------------------------
# Batch evaluation harness
# ---------------------------------------------------------------------

@dataclass
class QuestionEvalRecord:
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    em: float
    f1: float
    supporting_fact_recall: float
    k_used: int
    num_iterations: int
    latency_seconds: float
    generation_success: bool


@dataclass
class EvalSummary:
    num_questions: int
    avg_em: float
    avg_f1: float
    avg_supporting_fact_recall: float
    avg_k: float
    avg_iterations: float
    avg_latency_seconds: float
    generation_success_rate: float
    per_question: list[QuestionEvalRecord] = field(default_factory=list)


def evaluate_adaptive_pipeline(pipeline, questions: list) -> EvalSummary:
    """
    Run the adaptive pipeline over a list of QuestionRecords and compute
    the full metric set.
    """
    records = []
    for q in questions:
        result = pipeline.answer(q.question)
        record = QuestionEvalRecord(
            question_id=q.question_id,
            question=q.question,
            gold_answer=q.answer,
            predicted_answer=result.answer,
            em=exact_match_score(result.answer, q.answer),
            f1=f1_score(result.answer, q.answer),
            supporting_fact_recall=supporting_fact_title_recall(result.context_chunks, q.supporting_facts),
            k_used=result.final_k,
            num_iterations=result.num_iterations,
            latency_seconds=result.total_latency_seconds,
            generation_success=result.generation_success,
        )
        records.append(record)
        logger.info(
            "Eval[adaptive] qid=%s em=%.0f f1=%.2f sf_recall=%.2f k=%d iters=%d",
            record.question_id, record.em, record.f1, record.supporting_fact_recall,
            record.k_used, record.num_iterations,
        )

    return _summarize(records, has_iterations=True)


def evaluate_baseline_pipeline(pipeline, questions: list) -> EvalSummary:
    """
    Run the fixed Top-K baseline pipeline over the same question list and
    compute the same metric set (num_iterations is always 1 by definition).
    """
    records = []
    for q in questions:
        result = pipeline.answer(q.question)
        record = QuestionEvalRecord(
            question_id=q.question_id,
            question=q.question,
            gold_answer=q.answer,
            predicted_answer=result.answer,
            em=exact_match_score(result.answer, q.answer),
            f1=f1_score(result.answer, q.answer),
            supporting_fact_recall=supporting_fact_title_recall(result.context_chunks, q.supporting_facts),
            k_used=result.k,
            num_iterations=1,
            latency_seconds=result.total_latency_seconds,
            generation_success=result.generation_success,
        )
        records.append(record)
        logger.info(
            "Eval[baseline] qid=%s em=%.0f f1=%.2f sf_recall=%.2f k=%d",
            record.question_id, record.em, record.f1, record.supporting_fact_recall, record.k_used,
        )

    return _summarize(records, has_iterations=False)


def _summarize(records: list[QuestionEvalRecord], has_iterations: bool) -> EvalSummary:
    n = len(records)
    if n == 0:
        raise ValueError("Cannot summarize an empty evaluation set.")

    successful = [r for r in records if r.generation_success]

    return EvalSummary(
        num_questions=n,
        avg_em=sum(r.em for r in records) / n,
        avg_f1=sum(r.f1 for r in records) / n,
        avg_supporting_fact_recall=sum(r.supporting_fact_recall for r in records) / n,
        avg_k=sum(r.k_used for r in records) / n,
        avg_iterations=sum(r.num_iterations for r in records) / n if has_iterations else 1.0,
        avg_latency_seconds=sum(r.latency_seconds for r in records) / n,
        generation_success_rate=len(successful) / n,
        per_question=records,
    )


def print_comparison(adaptive_summary: EvalSummary, baseline_summary: EvalSummary) -> None:
    """Print a side-by-side adaptive vs. baseline comparison table."""
    print("\n=== Adaptive vs. Fixed Top-5 Baseline ===")
    print(f"{'Metric':<28}{'Adaptive':<15}{'Baseline (K=5)':<15}")
    print(f"{'Exact Match':<28}{adaptive_summary.avg_em:<15.3f}{baseline_summary.avg_em:<15.3f}")
    print(f"{'F1':<28}{adaptive_summary.avg_f1:<15.3f}{baseline_summary.avg_f1:<15.3f}")
    print(f"{'Supporting-fact recall':<28}{adaptive_summary.avg_supporting_fact_recall:<15.3f}{baseline_summary.avg_supporting_fact_recall:<15.3f}")
    print(f"{'Avg K used':<28}{adaptive_summary.avg_k:<15.2f}{baseline_summary.avg_k:<15.2f}")
    print(f"{'Avg iterations':<28}{adaptive_summary.avg_iterations:<15.2f}{baseline_summary.avg_iterations:<15.2f}")
    print(f"{'Avg latency (s)':<28}{adaptive_summary.avg_latency_seconds:<15.2f}{baseline_summary.avg_latency_seconds:<15.2f}")
    print(f"{'Generation success rate':<28}{adaptive_summary.generation_success_rate:<15.2%}{baseline_summary.generation_success_rate:<15.2%}")