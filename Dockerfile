FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# ✅ Add GL and X libs so OpenCV can import (fixes libGL.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .

# CRITICAL: Use gevent worker for streaming performance (workers=1 for low memory/high concurrency)
CMD ["/bin/sh", "-c", "gunicorn app:app -k gevent --timeout 180 --workers 1 --bind 0.0.0.0:${PORT:-10000}"]
