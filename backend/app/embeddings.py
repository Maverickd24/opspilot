"""
Thin wrapper around a local sentence-transformers model.

Why a local embedding model instead of an API: Groq (our LLM constraint)
doesn't currently serve an embeddings endpoint, and pulling in a second
paid provider just for embeddings adds another API key and another
failure point for a pilot. all-MiniLM-L6-v2 is small (~80MB), runs fine
on a free-tier CPU instance, and is good enough for this document scale.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns L2-normalized float32 embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, settings.EMBEDDING_DIM), dtype="float32")
    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return (vectors / norms).astype("float32")
