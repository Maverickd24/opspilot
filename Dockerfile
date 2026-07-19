FROM python:3.12-slim

WORKDIR /app

# System deps for faiss/torch wheels to build/run cleanly
# build-essential: needed to build any compiled deps during pip install
# libgomp1: faiss-cpu links against OpenMP at import time; python:3.12-slim
#           doesn't include it by default, which crashes the app on startup
#           with an ImportError that only surfaces in a truly minimal image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
