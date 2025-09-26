# Multi-stage Dockerfile for service_dataprep
# API service for track stitching verification workflows

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy shared libraries requirements first for better caching
COPY shared_libs/requirements.txt /app/shared_libs/requirements.txt

# Copy service-specific requirements
COPY services/service_dataprep/requirements.txt /app/services/service_dataprep/requirements.txt

# Install shared libraries dependencies first
RUN pip install --no-cache-dir -r shared_libs/requirements.txt

# Install service-specific dependencies
RUN pip install --no-cache-dir -r services/service_dataprep/requirements.txt

# Stage 3: Final application image
FROM dependencies AS application

# Copy shared libraries
COPY shared_libs /app/shared_libs

# Copy service code
COPY services/service_dataprep /app/services/service_dataprep

# Copy necessary root configuration files
COPY config.toml /app/config.toml

# Set PYTHONPATH globally for all users and entrypoints
ENV PYTHONPATH=/app/shared_libs:/app/services/service_dataprep/src \
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
CMD ["python", "-m", "services.service_dataprep.src.main"]