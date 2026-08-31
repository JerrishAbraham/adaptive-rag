"""
Module 1, Step 2 — Document Preprocessing.

Cleans DocumentRecords produced by data_loader.py without destroying
semantic information. Sentence boundaries (as provided by HotpotQA's
own sentence splits) are preserved intact — this stage only normalizes
whitespace/unicode artifacts and drops truly empty sentences. It does
NOT chunk, embed, or index anything.
"""

from __future__ import annotations

import copy
import logging
import re
import unicodedata

from src.data_loader import DocumentRecord

logger = logging.getLogger(__name__)


def clean_sentence(sentence: str) -> str:
    """
    Normalize a single sentence's text.

    - Unicode-normalize (NFKC) to fold odd Unicode variants of the same
      character (e.g. different apostrophe code points) into one form.
    - Collapse repeated whitespace/newlines/tabs into single spaces.
    - Strip leading/trailing whitespace.

    Sentence-level word content and order are never altered — only
    surface-level whitespace/encoding noise is removed.
    """
    if not sentence:
        return ""

    text = unicodedata.normalize("NFKC", sentence)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def clean_document(document: DocumentRecord) -> DocumentRecord:
    """
    Return a cleaned copy of a DocumentRecord.

    - Cleans each sentence individually via clean_sentence.
    - Drops sentences that become empty after cleaning (rare, but can
      happen with stray whitespace-only entries in the raw data).
    - Rebuilds `text` by re-joining the cleaned sentence list, so `text`
      and `sentences` stay consistent with each other.
    - Original DocumentRecord is not mutated; a copy is returned.
    """
    cleaned_sentences = [clean_sentence(s) for s in document.sentences]
    cleaned_sentences = [s for s in cleaned_sentences if s]  # drop empties

    dropped = len(document.sentences) - len(cleaned_sentences)
    if dropped > 0:
        logger.debug(
            "Document %s (%s): dropped %d empty sentence(s) after cleaning",
            document.document_id, document.title, dropped,
        )

    cleaned = copy.deepcopy(document)
    cleaned.sentences = cleaned_sentences
    cleaned.text = " ".join(cleaned_sentences)
    cleaned.metadata = {
        **document.metadata,
        "num_sentences": len(cleaned_sentences),
        "preprocessed": True,
    }
    return cleaned


def preprocess_documents(documents: list[DocumentRecord]) -> list[DocumentRecord]:
    """
    Clean a list of DocumentRecords and drop any that end up with no
    sentences at all (fully empty articles are not usable downstream).
    """
    cleaned_docs = []
    for doc in documents:
        cleaned = clean_document(doc)
        if not cleaned.sentences:
            logger.warning(
                "Document %s (%s) had no non-empty sentences after cleaning; dropping.",
                doc.document_id, doc.title,
            )
            continue
        cleaned_docs.append(cleaned)

    logger.info(
        "Preprocessed %d documents -> %d retained (%d dropped as empty)",
        len(documents), len(cleaned_docs), len(documents) - len(cleaned_docs),
    )
    return cleaned_docs