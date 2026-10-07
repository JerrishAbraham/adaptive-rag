

from __future__ import annotations

import copy
import logging
import re
import unicodedata

from src.data_loader import DocumentRecord

logger = logging.getLogger(__name__)


def clean_sentence(sentence: str) -> str:
    
    if not sentence:
        return ""

    text = unicodedata.normalize("NFKC", sentence)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def clean_document(document: DocumentRecord) -> DocumentRecord:
    
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