# WHAT THIS MODULE DOES:
# - Builds RAG prompt from context chunks and question.

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use the information from the CONTEXT sections to provide accurate, detailed answers. "
    "Always cite your sources using numbers in square brackets like [1]. "
    "If the context doesn't contain enough information to fully answer the question, "
    "provide what information you can from the context and mention what's missing."
)

def build_rag_prompt(chunks, user_question: str) -> str:
    pieces = [SYSTEM_PROMPT, "\n\nCONTEXT:\n"]
    for i, c in enumerate(chunks, start=1):
        src = c.get("metadata", {}).get("source_url") or c.get("metadata", {}).get("source_name") or "unknown"
        pieces.append(f"[{i}] (source: {src})\n{c.get('text')}\n")
    pieces.append("\nUSER QUESTION:\n" + user_question + "\n")
    pieces.append("\nINSTRUCTIONS:\nProvide a helpful answer based on the context above. Include citations like [1] for your sources. Be specific and detailed in your response.")
    return "\n\n".join(pieces)
