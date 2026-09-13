"""
Stage 4 — Prompt Construction.

Builds a controlled RAG prompt from the query and optimized (diversity-
filtered) context. Context and question are kept clearly separated so
the generator's instruction to "answer only from supplied context" has
an unambiguous boundary to follow.
"""

from __future__ import annotations

SYSTEM_INSTRUCTION = (
    "You are a question-answering assistant operating strictly over "
    "supplied context.\n"
    "Rules:\n"
    "1. Base your answer only on the CONTEXT provided below. Do not use "
    "outside knowledge.\n"
    "2. Do not invent, assume, or infer facts that are not supported by "
    "the CONTEXT.\n"
    "3. If the CONTEXT does not contain enough information to answer the "
    "QUESTION, explicitly say so instead of guessing.\n"
    "4. Answer concisely and directly."
)


def format_context(chunks: list[dict]) -> str:
    """
    Format a list of RetrievalResult-shaped dicts (must include 'title'
    and 'text') into a numbered context block, so individual chunks stay
    visually distinguishable in the prompt.
    """
    if not chunks:
        return "(no context retrieved)"

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] {chunk['title']}: {chunk['text']}")
    return "\n\n".join(lines)


def build_prompt(query: str, context_chunks: list[dict]) -> dict:
    """
    Build the full prompt as a dict with 'system' and 'user' keys, ready
    to hand to generator.generate(). Kept as a plain dict (not a raw
    string) so provider-specific message formatting stays inside
    generator.py, not here.
    """
    context_block = format_context(context_chunks)
    user_message = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"Answer the QUESTION using only the CONTEXT above."
    )
    return {"system": SYSTEM_INSTRUCTION, "user": user_message}