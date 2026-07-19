"""
Thin wrapper around the Groq chat completions API (OpenAI-compatible).
Two responsibilities, kept as two small functions rather than one big
prompt, so each is independently testable and explainable:

  1. rewrite_query   -- turns a follow-up like "and who does it apply to?"
                         into a standalone search query using chat history.
  2. generate_answer -- answers strictly from the retrieved chunks, with
                         inline [n] citations, or admits it doesn't know.
"""
from typing import List

from groq import Groq

from app.config import settings
from app.schemas import ChatTurn

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment (see .env.example)."
            )
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def rewrite_query(current_message: str, history: List[ChatTurn]) -> str:
    """Condense the current message + recent history into a standalone
    search query. If there's no history, this is a no-op."""
    if not history:
        return current_message

    recent = history[-6:]  # last 3 user/assistant pairs, roughly
    transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)

    system = (
        "You rewrite a follow-up chat message into a single standalone search "
        "query that makes sense without the conversation history. Resolve "
        "pronouns and implicit references (e.g. 'it', 'that clause', 'them') "
        "using the history. Output ONLY the rewritten query, nothing else. "
        "If the message is already standalone, return it unchanged."
    )
    user = f"Conversation so far:\n{transcript}\n\nFollow-up message: {current_message}\n\nStandalone query:"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else current_message
    except Exception:
        # Query rewriting is an enhancement, not critical path -- fall back
        # to the raw message rather than failing the whole chat turn.
        return current_message


def generate_answer(question: str, context_blocks: List[str], history: List[ChatTurn]) -> str:
    system = (
        "You are OpsPilot, an internal document assistant for a logistics "
        "company's operations team. Answer ONLY using the numbered context "
        "excerpts below. Every factual claim must be followed by a citation "
        "like [1] or [2] referencing the excerpt number it came from. "
        "If the answer is not contained in the excerpts, say clearly that "
        "you couldn't find it in the uploaded documents -- do not guess or "
        "use outside knowledge. Be concise and direct."
    )

    context_text = "\n\n".join(f"[{i+1}] {block}" for i, block in enumerate(context_blocks))

    messages = [{"role": "system", "content": system}]
    for turn in history[-6:]:
        role = "user" if turn.role == "user" else "assistant"
        messages.append({"role": role, "content": turn.content})

    messages.append(
        {
            "role": "user",
            "content": f"Context excerpts:\n\n{context_text}\n\nQuestion: {question}",
        }
    )

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()
