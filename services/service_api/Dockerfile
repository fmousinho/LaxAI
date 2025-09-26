# Multi-stage Dockerfile for service_api
# Lightweight API service for job queuing

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SERVICE_DATAPREP_NAME=laxai-service-dataprep \
    SERVICE_DATAPREP_PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy shared libraries requirements first for better caching
COPY shared_libs/requirements.txt /app/shared_libs/requirements.txt

# Copy service-specific requirements
COPY services/service_api/requirements.txt /app/services/service_api/requirements.txt

# Install shared libraries dependencies first
RUN pip install --no-cache-dir -r shared_libs/requirements.txt

# Install service-specific dependencies
RUN pip install --no-cache-dir -r services/service_api/requirements.txt

# Stage 3: Application code
FROM dependencies AS application

# Copy shared libraries
COPY shared_libs/ /app/shared_libs/

# Copy service API source code
COPY services/service_api/src/ /app/services/service_api/src/

# Copy necessary root configuration files
COPY config.toml /app/config.toml

# Set PYTHONPATH globally for all users and entrypoints
ENV PYTHONPATH=/app/shared_libs:/app/services/service_api/src \
    CONFIG_TOML_PATH=/app/config.toml

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Working directory for the service
WORKDIR /app/

# Default command - can be overridden
CMD ["uvicorn", "services.service_api.src.main:app", "--host", "0.0.0.0", "--port", "8080"]

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1