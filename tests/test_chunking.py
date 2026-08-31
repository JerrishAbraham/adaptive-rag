"""
Module 1, Step 3 — validation test for chunking.

Loads a tiny HotpotQA sample, preprocesses it, chunks it using config
values, and inspects chunk boundaries, sizes, and overlap behavior.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_hotpotqa_sample  # noqa: E402
from src.preprocessing import preprocess_documents  # noqa: E402
from src.chunking import chunk_documents, load_chunking_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_chunk_tiny_sample():
    documents, questions = load_hotpotqa_sample(
        split="validation", sample_limit=5, config="distractor"
    )
    cleaned_docs = preprocess_documents(documents)

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    chunking_cfg = load_chunking_config(str(config_path))
    chunk_size = chunking_cfg["chunk_size"]
    chunk_overlap = chunking_cfg["chunk_overlap"]

    print(f"\nUsing chunk_size={chunk_size}, chunk_overlap={chunk_overlap} from config.yaml")

    chunks = chunk_documents(cleaned_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 0, "No chunks were produced."
    assert len(chunks) >= len(cleaned_docs), "Expected at least one chunk per document."

    # Every chunk must trace back to a real document.
    doc_ids = {d.document_id for d in cleaned_docs}
    for c in chunks:
        assert c.document_id in doc_ids
        assert c.text.strip() != ""
        assert c.metadata["char_length"] == len(c.text)

    # chunk_index must be sequential per document, starting at 0.
    from collections import defaultdict
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c.document_id].append(c)
    for doc_id, doc_chunks in by_doc.items():
        indices = [c.chunk_index for c in sorted(doc_chunks, key=lambda c: c.chunk_index)]
        assert indices == list(range(len(indices))), f"Non-sequential chunk_index for {doc_id}"

    print("\n=== Summary ===")
    print(f"Documents: {len(cleaned_docs)}")
    print(f"Total chunks: {len(chunks)}")
    avg_len = sum(c.metadata["char_length"] for c in chunks) / len(chunks)
    print(f"Average chunk char length: {avg_len:.1f}")

    # Find a document that produced multiple chunks to inspect overlap.
    multi_chunk_doc_id = next((doc_id for doc_id, cs in by_doc.items() if len(cs) > 1), None)

    if multi_chunk_doc_id:
        print(f"\n=== Multi-chunk document example: {multi_chunk_doc_id} ===")
        doc_chunks = sorted(by_doc[multi_chunk_doc_id], key=lambda c: c.chunk_index)
        for c in doc_chunks:
            print(f"  chunk_index={c.chunk_index} chunk_id={c.chunk_id} "
                  f"len={c.metadata['char_length']} sentences={c.metadata['num_sentences']}")
        print(f"\n  chunk[0] tail: ...{doc_chunks[0].text[-100:]!r}")
        print(f"  chunk[1] head: {doc_chunks[1].text[:100]!r}...")
        print("  (tail/head above should visibly share overlapping sentence content)")
    else:
        print("\nNo multi-chunk document in this tiny sample (all articles fit in one chunk) "
              "— this is expected with only 5 questions; try a larger sample_limit to see overlap.")

    print("\nOK: chunking test passed.")


if __name__ == "__main__":
    test_chunk_tiny_sample()