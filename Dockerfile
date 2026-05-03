FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all source code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "src.cli", "--help"]
