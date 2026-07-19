"""
Embeddings via Google's Gemini embedding API (gemini-embedding-001), not a
local model.

History: this pilot first used sentence-transformers/torch locally, which
exceeded Render free tier's 512MB RAM ceiling and crashed the container.
Switching to fastembed (ONNX, no torch) reduced the footprint but a
large real-world PDF still pushed total memory (FastAPI + onnxruntime +
model weights + PDF processing, all in one 512MB box) over the limit.

Moving embedding computation to a hosted API removes that memory
category from the equation almost entirely -- the server now just makes
small HTTP calls instead of loading and running a model in-process. This
is within the assignment's "free-tier LLM APIs" constraint (Gemini has a
free embeddings tier), the same way Groq is used for chat.

gemini-embedding-001's native output is 3072 dimensions. We initially
tried requesting a truncated 768-dim output via outputDimensionality to
keep the FAISS index smaller, but the batchEmbedContents endpoint
returned full 3072-dim vectors regardless, which crashed FAISS with a
dimension-mismatch assertion. Using the native 3072 dimension everywhere
avoids depending on that truncation behaving consistently, at the cost
of a slightly larger (still trivial, for a session-scoped pilot) index.
"""
import httpx
import numpy as np

from app.config import settings

GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
_BATCH_SIZE = 90  # stay under Gemini's per-request batch limit


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment (see .env.example)."
        )

    url = GEMINI_EMBED_URL.format(model=settings.EMBEDDING_MODEL)
    payload = {
        "requests": [
            {
                "model": f"models/{settings.EMBEDDING_MODEL}",
                "content": {"parts": [{"text": text}]},
                "embedContentConfig": {"taskType": task_type},
            }
            for text in texts
        ]
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            url,
            headers={"x-goog-api-key": settings.GEMINI_API_KEY},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    return [item["values"] for item in data["embeddings"]]


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Returns L2-normalized float32 embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, settings.EMBEDDING_DIM), dtype="float32")

    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        all_vectors.extend(_embed_batch(batch, task_type))

    vectors = np.array(all_vectors, dtype="float32")

    if vectors.shape[1] != settings.EMBEDDING_DIM:
        raise RuntimeError(
            f"Gemini returned {vectors.shape[1]}-dim embeddings but EMBEDDING_DIM is "
            f"configured as {settings.EMBEDDING_DIM}. Update settings.EMBEDDING_DIM to match."
        )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return (vectors / norms).astype("float32")
