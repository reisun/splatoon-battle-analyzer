FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV headless and ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all source code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Run as non-root user
RUN useradd -m appuser && \
    mkdir -p /app/.ruff_cache /app/.pytest_cache && \
    chown -R appuser:appuser /app/.ruff_cache /app/.pytest_cache
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
