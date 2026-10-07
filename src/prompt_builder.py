

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
    
    if not chunks:
        return "(no context retrieved)"

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] {chunk['title']}: {chunk['text']}")
    return "\n\n".join(lines)


def build_prompt(query: str, context_chunks: list[dict]) -> dict:
    
    context_block = format_context(context_chunks)
    user_message = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"Answer the QUESTION using only the CONTEXT above."
    )
    return {"system": SYSTEM_INSTRUCTION, "user": user_message}