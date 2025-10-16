# Multi-stage Dockerfile for service_sticher
# API service for track stitching verification workflows

FROM python:3.12-slim AS base

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

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy shared libraries requirements first for better caching
COPY shared_libs/requirements.txt /app/shared_libs/requirements.txt

# Copy service-specific requirements
COPY services/service_stitcher/requirements.txt /app/services/service_stitcher/requirements.txt

# Install shared libraries dependencies first (with retry logic)
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt || \
    (echo "First attempt failed, retrying..." && sleep 5 && pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt) || \
    (echo "Second attempt failed, final retry..." && sleep 10 && pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt)

# Install service-specific dependencies (with retry logic)
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_stitcher/requirements.txt || \
    (echo "First attempt failed, retrying..." && sleep 5 && pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_stitcher/requirements.txt) || \
    (echo "Second attempt failed, final retry..." && sleep 10 && pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_stitcher/requirements.txt)

# Stage 3: Final application image
FROM dependencies AS application

# Copy shared libraries
COPY shared_libs /app/shared_libs

# Copy service code
COPY services/service_stitcher /app/services/service_stitcher

# Copy necessary root configuration files
COPY config.toml /app/config.toml

# Set PYTHONPATH globally for all users and entrypoints
ENV PYTHONPATH=/app/shared_libs:/app/services/service_stitcher/src \
    CONFIG_TOML_PATH=/app/config.toml

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "services.service_stitcher.src.main"]