"""
Central place for all environment-driven configuration.
Nothing secret is hardcoded here -- everything comes from the environment,
which on Render is set via the dashboard's Environment tab and locally via
a .env file (see .env.example at the repo root).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM (Groq) ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # --- Embeddings (Gemini API, hosted -- not run locally) ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    EMBEDDING_DIM: int = 768

    # --- Chunking ---
    CHUNK_SIZE_CHARS: int = int(os.getenv("CHUNK_SIZE_CHARS", "1000"))
    CHUNK_OVERLAP_CHARS: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

    # --- Retrieval ---
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # --- Uploads ---
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
    MAX_FILES_PER_UPLOAD: int = int(os.getenv("MAX_FILES_PER_UPLOAD", "5"))

    # --- CORS ---
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")


settings = Settings()

if not settings.GROQ_API_KEY:
    # We don't crash on import (so the app can still serve health checks /
    # frontend), but every LLM call will fail fast with a clear message.
    print("WARNING: GROQ_API_KEY is not set. Chat requests will fail until it is.")

if not settings.GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Document upload/embedding will fail until it is.")
