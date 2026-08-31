"""
Module 1, Step 2 — validation test for document preprocessing.

Loads a tiny HotpotQA sample, preprocesses the resulting documents, and
prints before/after comparisons so cleaning behavior can be manually
inspected.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_hotpotqa_sample  # noqa: E402
from src.preprocessing import preprocess_documents  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_preprocess_tiny_sample():
    documents, questions = load_hotpotqa_sample(
        split="validation", sample_limit=5, config="distractor"
    )
    assert len(documents) > 0

    cleaned_docs = preprocess_documents(documents)

    assert len(cleaned_docs) > 0, "All documents were dropped during preprocessing."
    assert len(cleaned_docs) <= len(documents)

    for doc in cleaned_docs:
        assert doc.text == " ".join(doc.sentences)
        assert all(s == s.strip() for s in doc.sentences)
        assert doc.metadata.get("preprocessed") is True

    print("\n=== Summary ===")
    print(f"Documents before preprocessing: {len(documents)}")
    print(f"Documents after preprocessing:  {len(cleaned_docs)}")

    print("\n=== Before / After (first document) ===")
    raw_doc = documents[0]
    clean_doc = next(d for d in cleaned_docs if d.document_id == raw_doc.document_id)

    print(f"title: {raw_doc.title}")
    print(f"BEFORE sentences[0]: {raw_doc.sentences[0]!r}")
    print(f"AFTER  sentences[0]: {clean_doc.sentences[0]!r}")
    print(f"BEFORE num_sentences: {len(raw_doc.sentences)}")
    print(f"AFTER  num_sentences: {len(clean_doc.sentences)}")
    print(f"AFTER text[:200]: {clean_doc.text[:200]!r}")

    print("\nOK: preprocessing test passed.")


if __name__ == "__main__":
    test_preprocess_tiny_sample()