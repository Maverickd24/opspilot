"""
Keeps two things per session_id, both in memory:
  - the list of documents that have been uploaded (for the "which
    documents are loaded" UI requirement)
  - the running chat history (for conversational memory)

Same in-memory trade-off as vector_store.py -- see the note there.
"""
import threading
from dataclasses import dataclass, field
from typing import Dict, List

from app.schemas import ChatTurn, DocumentInfo


@dataclass
class Session:
    documents: List[DocumentInfo] = field(default_factory=list)
    history: List[ChatTurn] = field(default_factory=list)


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session()
            return self._sessions[session_id]

    def add_documents(self, session_id: str, docs: List[DocumentInfo]) -> None:
        session = self.get_or_create(session_id)
        with self._lock:
            session.documents.extend(docs)

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        session = self.get_or_create(session_id)
        with self._lock:
            session.history.append(ChatTurn(role=role, content=content))
            # Cap history to the last 20 turns so prompts don't grow unbounded.
            if len(session.history) > 20:
                session.history = session.history[-20:]

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions


session_store = SessionStore()
