"""
Module 1, Step 1 — HotpotQA Data Loader.

Loads the HotpotQA 'distractor' config, extracts context articles into
deduplicated DocumentRecords, and keeps question/answer/supporting-fact
evaluation data separate as QuestionRecords.

No embeddings, chunking, or indexing happens here — this stage only
produces a clean internal representation of the raw dataset.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """A single deduplicated Wikipedia-style article from HotpotQA context."""
    document_id: str
    title: str
    text: str
    sentences: list[str]
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class QuestionRecord:
    """A single HotpotQA benchmark question with its evaluation metadata."""
    question_id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: list[tuple[str, int]]
    context_titles: list[str]
    split: str


def normalize_title(title: str) -> str:
    """Normalize a title for deduplication (lowercase, collapse whitespace)."""
    return re.sub(r"\s+", " ", title.strip().lower())


def load_hotpotqa_raw(
    split: str = "validation",
    sample_limit: Optional[int] = None,
    config: str = "distractor",
):
    """
    Load the raw HotpotQA dataset via Hugging Face `datasets`.

    Args:
        split: 'train' or 'validation'.
        sample_limit: if set, only load this many examples (for fast dev/testing).
        config: HotpotQA config name; locked to 'distractor' per project scope.

    Returns:
        A Hugging Face Dataset object.
    """
    from datasets import load_dataset

    logger.info("Loading HotpotQA config=%s split=%s sample_limit=%s", config, split, sample_limit)
    # Note: HF moved this dataset under the 'hotpotqa/' namespace; the bare
    # 'hotpot_qa' id triggers a broken legacy-script resolution path in
    # newer huggingface_hub versions.
    dataset = load_dataset("hotpotqa/hotpot_qa", config, split=split)

    if sample_limit is not None:
        sample_limit = min(sample_limit, len(dataset))
        dataset = dataset.select(range(sample_limit))

    logger.info("Loaded %d raw examples", len(dataset))
    return dataset


def build_records(raw_dataset, split: str) -> tuple[list[DocumentRecord], list[QuestionRecord]]:
    """
    Convert raw HotpotQA examples into deduplicated DocumentRecords and
    separate QuestionRecords.

    Deduplication is by normalized article title: the same article appears
    as context for many different questions, and we only want to store it once.
    """
    documents: dict[str, DocumentRecord] = {}
    questions: list[QuestionRecord] = []

    for example in raw_dataset:
        titles = example["context"]["title"]
        sentences_per_article = example["context"]["sentences"]
        context_titles_for_this_question = []

        for title, sentences in zip(titles, sentences_per_article):
            context_titles_for_this_question.append(title)
            norm = normalize_title(title)

            if norm not in documents:
                doc_id = f"doc_{len(documents):06d}"
                documents[norm] = DocumentRecord(
                    document_id=doc_id,
                    title=title,
                    text=" ".join(sentences),
                    sentences=list(sentences),
                    source="hotpotqa_context",
                    metadata={"num_sentences": len(sentences), "split": split},
                )

        supporting_facts = list(
            zip(
                example["supporting_facts"]["title"],
                example["supporting_facts"]["sent_id"],
            )
        )

        questions.append(
            QuestionRecord(
                question_id=example["id"],
                question=example["question"],
                answer=example.get("answer", ""),
                type=example.get("type", ""),
                level=example.get("level", ""),
                supporting_facts=supporting_facts,
                context_titles=context_titles_for_this_question,
                split=split,
            )
        )

    logger.info(
        "Built %d unique documents and %d question records from %d raw examples",
        len(documents), len(questions), len(raw_dataset),
    )
    return list(documents.values()), questions


def load_hotpotqa_sample(
    split: str = "validation",
    sample_limit: int = 50,
    config: str = "distractor",
) -> tuple[list[DocumentRecord], list[QuestionRecord]]:
    """
    Convenience entry point for Module 1, Step 1.

    Loads a small HotpotQA sample and returns (documents, questions).
    This is the function later stages (chunking, embeddings) will build on.
    """
    raw = load_hotpotqa_raw(split=split, sample_limit=sample_limit, config=config)
    return build_records(raw, split=split)