FROM python:3.11-slim

# Optional: nicer logs & faster installs
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# System deps that help manylinux wheels (opencv/mediapipe) behave well
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgcc1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Use Render's PORT if provided, else 10000 locally
CMD ["/bin/sh", "-c", "gunicorn app:app --timeout 180 --workers 1 --threads 8 --bind 0.0.0.0:${PORT:-10000}"]
