from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int


class UploadResponse(BaseModel):
    session_id: str
    documents: List[DocumentInfo]


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=2000)


class Citation(BaseModel):
    doc_id: str
    filename: str
    page: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    rewritten_query: Optional[str] = None


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class SessionState(BaseModel):
    session_id: str
    documents: List[DocumentInfo]
    history: List[ChatTurn]
