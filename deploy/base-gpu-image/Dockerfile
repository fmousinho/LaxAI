# Base GPU Image for Service Training
# Includes system dependencies and Python packages, but NO application code.

FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3-opencv \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements files
# Assuming build context is root of repo
COPY shared_libs/requirements.txt /app/shared_libs/requirements.txt
COPY services/service_training/requirements.txt /app/services/service_training/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt && \
    pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_training/requirements.txt

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /home/appuser/.config/matplotlib && \
    mkdir -p /home/appuser/.cache/fontconfig && \
    chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /app

# Set PYTHONPATH (useful for when app code is mounted/added later)
ENV PYTHONPATH=/app:/app/services/service_training/src

# Switch to non-root user
USER appuser
