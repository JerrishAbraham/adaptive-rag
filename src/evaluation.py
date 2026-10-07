

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)



def normalize_answer(text: str) -> str:
    
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



def supporting_fact_title_recall(
    context_chunks: list[dict],
    supporting_facts: list[tuple[str, int]],
) -> float:
 
    gold_titles = {t for t, _ in supporting_facts}

    if not gold_titles:
        return 1.0

    if not context_chunks:
        return 0.0

    retrieved_titles = {c["title"] for c in context_chunks}
    hit = gold_titles & retrieved_titles

    return len(hit) / len(gold_titles)


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

    # Adaptive retrieval diagnostics.
    # These remain None/{} for the fixed baseline.
    confidence_score: float | None = None
    sufficient: bool | None = None
    quality_metrics: dict = field(default_factory=dict)


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
            supporting_fact_recall=supporting_fact_title_recall(
                result.context_chunks,
                q.supporting_facts,
            ),
            k_used=result.final_k,
            num_iterations=result.num_iterations,
            latency_seconds=result.total_latency_seconds,
            generation_success=result.generation_success,

            # New adaptive diagnostics.
            confidence_score=result.confidence_score,
            sufficient=result.sufficient,
            quality_metrics=result.quality_metrics,
        )

        records.append(record)

        logger.info(
            "Eval[adaptive] qid=%s em=%.0f f1=%.2f "
            "sf_recall=%.2f k=%d iters=%d confidence=%s sufficient=%s",
            record.question_id,
            record.em,
            record.f1,
            record.supporting_fact_recall,
            record.k_used,
            record.num_iterations,
            (
                f"{record.confidence_score:.4f}"
                if record.confidence_score is not None
                else "None"
            ),
            record.sufficient,
        )

    return _summarize(records, has_iterations=True)


def evaluate_baseline_pipeline(pipeline, questions: list) -> EvalSummary:
   
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
            supporting_fact_recall=supporting_fact_title_recall(
                result.context_chunks,
                q.supporting_facts,
            ),
            k_used=result.k,
            num_iterations=1,
            latency_seconds=result.total_latency_seconds,
            generation_success=result.generation_success,

            # Baseline has no adaptive quality assessment.
            confidence_score=result.confidence_score,
            sufficient=result.sufficient,
            quality_metrics=result.quality_metrics,
        )

        records.append(record)

        logger.info(
            "Eval[baseline] qid=%s em=%.0f f1=%.2f "
            "sf_recall=%.2f k=%d",
            record.question_id,
            record.em,
            record.f1,
            record.supporting_fact_recall,
            record.k_used,
        )

    return _summarize(records, has_iterations=False)


def _summarize(
    records: list[QuestionEvalRecord],
    has_iterations: bool,
) -> EvalSummary:
    n = len(records)

    if n == 0:
        raise ValueError("Cannot summarize an empty evaluation set.")

    successful = [r for r in records if r.generation_success]

    return EvalSummary(
        num_questions=n,
        avg_em=sum(r.em for r in records) / n,
        avg_f1=sum(r.f1 for r in records) / n,
        avg_supporting_fact_recall=(
            sum(r.supporting_fact_recall for r in records) / n
        ),
        avg_k=sum(r.k_used for r in records) / n,
        avg_iterations=(
            sum(r.num_iterations for r in records) / n
            if has_iterations
            else 1.0
        ),
        avg_latency_seconds=sum(r.latency_seconds for r in records) / n,
        generation_success_rate=len(successful) / n,
        per_question=records,
    )


def print_comparison(
    adaptive_summary: EvalSummary,
    baseline_summary: EvalSummary,
) -> None:
    """Print a side-by-side adaptive vs. baseline comparison table."""
    print("\n=== Adaptive vs. Fixed Top-5 Baseline ===")
    print(f"{'Metric':<28}{'Adaptive':<15}{'Baseline (K=5)':<15}")
    print(
        f"{'Exact Match':<28}"
        f"{adaptive_summary.avg_em:<15.3f}"
        f"{baseline_summary.avg_em:<15.3f}"
    )
    print(
        f"{'F1':<28}"
        f"{adaptive_summary.avg_f1:<15.3f}"
        f"{baseline_summary.avg_f1:<15.3f}"
    )
    print(
        f"{'Supporting-fact recall':<28}"
        f"{adaptive_summary.avg_supporting_fact_recall:<15.3f}"
        f"{baseline_summary.avg_supporting_fact_recall:<15.3f}"
    )
    print(
        f"{'Avg K used':<28}"
        f"{adaptive_summary.avg_k:<15.2f}"
        f"{baseline_summary.avg_k:<15.2f}"
    )
    print(
        f"{'Avg iterations':<28}"
        f"{adaptive_summary.avg_iterations:<15.2f}"
        f"{baseline_summary.avg_iterations:<15.2f}"
    )
    print(
        f"{'Avg latency (s)':<28}"
        f"{adaptive_summary.avg_latency_seconds:<15.2f}"
        f"{baseline_summary.avg_latency_seconds:<15.2f}"
    )
    print(
        f"{'Generation success rate':<28}"
        f"{adaptive_summary.generation_success_rate:<15.2%}"
        f"{baseline_summary.generation_success_rate:<15.2%}"
    )