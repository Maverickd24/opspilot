"""
One FAISS index per chat session, kept in memory.

This is a deliberate pilot-scope trade-off, called out in the README:
free-tier hosts (Render free web services) spin the container down on
idle and wipe local disk, so persisting to disk wouldn't survive
restarts anyway without adding a paid volume or an external vector DB.
For a pilot where a user uploads docs and chats in one sitting, an
in-memory index keyed by session_id is simple, fast, and easy to reason
about. Swapping this module for Chroma/pgvector/Pinecone later is a
contained change -- nothing outside this file needs to know.
"""
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import faiss
import numpy as np

from app.config import settings
from app.pdf_processor import Chunk


@dataclass
class SessionIndex:
    index: faiss.Index
    chunks: List[Chunk] = field(default_factory=list)


class VectorStore:
    def __init__(self):
        self._sessions: Dict[str, SessionIndex] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, session_id: str) -> SessionIndex:
        if session_id not in self._sessions:
            index = faiss.IndexFlatIP(settings.EMBEDDING_DIM)
            self._sessions[session_id] = SessionIndex(index=index)
        return self._sessions[session_id]

    def add_chunks(self, session_id: str, chunks: List[Chunk], vectors: np.ndarray) -> None:
        with self._lock:
            session = self._get_or_create(session_id)
            if len(chunks) == 0:
                return
            session.index.add(vectors)
            session.chunks.extend(chunks)

    def search(self, session_id: str, query_vector: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        with self._lock:
            if session_id not in self._sessions:
                return []
            session = self._sessions[session_id]
            if session.index.ntotal == 0:
                return []
            k = min(top_k, session.index.ntotal)
            scores, indices = session.index.search(query_vector.reshape(1, -1), k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((session.chunks[idx], float(score)))
            return results

    def has_documents(self, session_id: str) -> bool:
        return session_id in self._sessions and self._sessions[session_id].index.ntotal > 0

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


vector_store = VectorStore()
