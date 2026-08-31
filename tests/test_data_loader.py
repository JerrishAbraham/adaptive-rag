"""
Module 1, Step 1 — validation test for the HotpotQA data loader.

Loads a tiny sample and prints document/question statistics so results
can be manually inspected before moving on to chunking/embeddings.
"""

import logging
import sys
from pathlib import Path

# Allow running this file directly (python tests/test_data_loader.py)
# without needing the package installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_hotpotqa_sample  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_load_tiny_sample():
    documents, questions = load_hotpotqa_sample(
        split="validation", sample_limit=5, config="distractor"
    )

    assert len(documents) > 0, "No documents were extracted."
    assert len(questions) == 5, "Expected exactly 5 question records."

    for q in questions:
        assert q.question_id
        assert q.question
        assert isinstance(q.supporting_facts, list)

    print("\n=== Summary ===")
    print(f"Raw questions loaded: {len(questions)}")
    print(f"Unique documents extracted: {len(documents)}")

    avg_sentences = sum(len(d.sentences) for d in documents) / len(documents)
    print(f"Average sentences per document: {avg_sentences:.2f}")

    print("\n=== Sample document ===")
    d0 = documents[0]
    print(f"document_id: {d0.document_id}")
    print(f"title:       {d0.title}")
    print(f"sentences:   {len(d0.sentences)}")
    print(f"text[:200]:  {d0.text[:200]!r}")

    print("\n=== Sample questions ===")
    for q in questions[:3]:
        print(f"- id={q.question_id} type={q.type} level={q.level}")
        print(f"  question: {q.question}")
        print(f"  answer:   {q.answer}")
        print(f"  supporting_facts: {q.supporting_facts[:2]} ...")
        print(f"  context_titles: {q.context_titles}")

    print("\nOK: data loader test passed.")


if __name__ == "__main__":
    test_load_tiny_sample()