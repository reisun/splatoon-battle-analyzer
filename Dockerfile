FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV headless, ffmpeg, and Node.js (for Claude CLI)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
        curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Copy all source code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Run as non-root user (required for claude --dangerously-skip-permissions)
RUN useradd -m appuser && \
    mkdir -p /app/.ruff_cache /app/.pytest_cache && \
    chown -R appuser:appuser /app/.ruff_cache /app/.pytest_cache
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
