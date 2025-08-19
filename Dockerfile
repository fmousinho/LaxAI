ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Default base is a lightweight Python slim image (CPU). To use a different
# base (for example a PyTorch CUDA runtime on GPU-enabled Linux hosts), pass
# --build-arg BASE_IMAGE=<image> when building.

# Keep container output unbuffered and avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level build dependencies for typical Python wheels and CA roots
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
         build-essential gcc curl \
         ffmpeg \
     && rm -rf /var/lib/apt/lists/*

# Copy package metadata first to leverage Docker layer caching
COPY pyproject.toml requirements.txt /app/

# Note: requirements.txt now prefers opencv-python-headless to avoid GUI deps

# Copy source, docs and config files explicitly so they are available in the image
# (adjust these paths if you use a different layout)
COPY src /app/src
COPY documentation /app/documentation
COPY config.toml /app/

# Upgrade pip and install build tools and the package. Installing the package via
# `pip install .` builds and installs the wheel into the environment so the
# console scripts / entrypoints become available.
RUN python -m pip install --upgrade pip setuptools wheel build \
    && if [ -f /app/requirements.txt ]; then \
        pip install --no-cache-dir -r /app/requirements.txt; \
    fi \
    && pip install --no-cache-dir /app

# Ensure our application code is importable from /app/src
ENV PYTHONPATH=/app/src

# Create a non-root user for runtime and give them ownership of /app
RUN useradd -m -u 1000 laxai \
 && chown -R laxai:laxai /app

# Switch to non-root user for running the service
USER laxai
ENV HOME=/home/laxai

EXPOSE 8000

# Default command: run uvicorn. For development, you can override to add --reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
