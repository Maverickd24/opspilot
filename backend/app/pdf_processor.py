"""
Turns a raw PDF into a list of Chunk objects ready for embedding.

Chunking strategy (the thing the assignment says we'll be asked about):
  1. Extract text page-by-page (never across page boundaries) so every
     chunk can be traced back to a single, correct page number for citation.
  2. Within a page, split on blank lines first (paragraph-aware), then
     greedily pack paragraphs into ~CHUNK_SIZE_CHARS windows so we don't
     cut a sentence in half more than necessary.
  3. If a single paragraph is longer than the chunk size (e.g. a dense
     clause block in a contract), hard-split it with a character-based
     sliding window and CHUNK_OVERLAP_CHARS of overlap, so a clause that
     straddles a split point still appears whole in at least one chunk.

This is a deliberately simple, dependency-free splitter (no LangChain)
so every line is something we can explain and change in the walkthrough.
"""
import re
import uuid
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader

from app.config import settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str


@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    page_count: int
    chunks: List[Chunk] = field(default_factory=list)


def _split_paragraph(paragraph: str, size: int, overlap: int) -> List[str]:
    """Hard character-window split for a single oversized paragraph."""
    if len(paragraph) <= size:
        return [paragraph]
    pieces = []
    start = 0
    while start < len(paragraph):
        end = min(start + size, len(paragraph))
        pieces.append(paragraph[start:end])
        if end == len(paragraph):
            break
        start = end - overlap
    return pieces


def _chunk_page_text(text: str, size: int, overlap: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        # Oversized paragraph: flush current buffer, then hard-split it.
        if len(para) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_paragraph(para, size, overlap))
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    return chunks


def process_pdf(file_bytes: bytes, filename: str) -> ProcessedDocument:
    reader = PdfReader.__new__(PdfReader)  # placeholder to keep type checkers calm
    import io

    reader = PdfReader(io.BytesIO(file_bytes))
    doc_id = str(uuid.uuid4())[:8]

    doc = ProcessedDocument(doc_id=doc_id, filename=filename, page_count=len(reader.pages))

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        for piece in _chunk_page_text(raw_text, settings.CHUNK_SIZE_CHARS, settings.CHUNK_OVERLAP_CHARS):
            doc.chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{page_number}-{len(doc.chunks)}",
                    doc_id=doc_id,
                    filename=filename,
                    page=page_number,
                    text=piece,
                )
            )

    return doc
