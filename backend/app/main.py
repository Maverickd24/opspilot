import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import rag_service
from app.config import settings
from app.embeddings import embed_texts
from app.pdf_processor import process_pdf
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentInfo,
    SessionState,
    UploadResponse,
)
from app.session_store import session_store
from app.vector_store import vector_store

app = FastAPI(title="OpsPilot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/sessions")
def create_session():
    session_id = str(uuid.uuid4())
    session_store.get_or_create(session_id)
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}", response_model=SessionState)
def get_session(session_id: str):
    if not session_store.exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    session = session_store.get_or_create(session_id)
    return SessionState(session_id=session_id, documents=session.documents, history=session.history)


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_documents(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > settings.MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max {settings.MAX_FILES_PER_UPLOAD} per upload.",
        )

    session_store.get_or_create(session_id)
    added_docs: List[DocumentInfo] = []

    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"'{upload.filename}' is not a PDF.")

        file_bytes = await upload.read()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"'{upload.filename}' is {size_mb:.1f}MB, exceeds {settings.MAX_FILE_SIZE_MB}MB limit.",
            )

        try:
            processed = process_pdf(file_bytes, upload.filename)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not read '{upload.filename}'. Is it a valid, non-encrypted PDF? ({exc})",
            )

        if not processed.chunks:
            raise HTTPException(
                status_code=422,
                detail=f"No extractable text found in '{upload.filename}'. "
                f"It may be a scanned image PDF without OCR text.",
            )

        texts = [c.text for c in processed.chunks]
        try:
            vectors = embed_texts(texts)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Embedding service is unavailable right now, please retry in a moment. ({exc})",
            )
        vector_store.add_chunks(session_id, processed.chunks, vectors)

        doc_info = DocumentInfo(
            doc_id=processed.doc_id,
            filename=processed.filename,
            page_count=processed.page_count,
            chunk_count=len(processed.chunks),
        )
        added_docs.append(doc_info)

    session_store.add_documents(session_id, added_docs)
    session = session_store.get_or_create(session_id)
    return UploadResponse(session_id=session_id, documents=session.documents)


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not session_store.exists(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Start a new session and upload documents first.")

    if not vector_store.has_documents(request.session_id):
        raise HTTPException(
            status_code=400,
            detail="No documents loaded in this session yet. Upload at least one PDF before chatting.",
        )

    session = session_store.get_or_create(request.session_id)

    try:
        answer, citations, rewritten = rag_service.answer_question(
            request.session_id, request.message, session.history
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"The assistant failed to respond: {exc}")

    session_store.add_turn(request.session_id, "user", request.message)
    session_store.add_turn(request.session_id, "assistant", answer)

    return ChatResponse(answer=answer, citations=citations, rewritten_query=rewritten)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    vector_store.clear_session(session_id)
    return {"status": "cleared"}


# --- Serve the frontend as static files (single deployable service) ---
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
