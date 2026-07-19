from typing import List

from app import llm_service
from app.config import settings
from app.embeddings import embed_texts
from app.schemas import ChatTurn, Citation
from app.vector_store import vector_store


def answer_question(session_id: str, question: str, history: List[ChatTurn]) -> tuple[str, List[Citation], str]:
    rewritten = llm_service.rewrite_query(question, history)

    query_vector = embed_texts([rewritten])[0]
    results = vector_store.search(session_id, query_vector, settings.TOP_K)

    if not results:
        return (
            "I don't have any relevant content to answer that yet. "
            "Please upload the documents you'd like me to search first.",
            [],
            rewritten,
        )

    context_blocks = [chunk.text for chunk, _score in results]
    answer = llm_service.generate_answer(question, context_blocks, history)

    citations = [
        Citation(
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            page=chunk.page,
            snippet=(chunk.text[:220] + "...") if len(chunk.text) > 220 else chunk.text,
        )
        for chunk, _score in results
    ]

    return answer, citations, rewritten
