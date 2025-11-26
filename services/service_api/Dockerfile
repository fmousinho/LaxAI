# Multi-stage Dockerfile for service_api
# Lightweight API service for job queuing

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    SERVICE_DATAPREP_NAME=laxai-service-dataprep \
    SERVICE_DATAPREP_PORT=8080 \
    STITCHER_SERVICE_URL=https://laxai-service-stitcher-kfccnooita-uc.a.run.app

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

# Install shared libraries dependencies first (with retry logic)
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt || \
    (echo "First attempt failed, retrying..." && sleep 5 && pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt) || \
    (echo "Second attempt failed, final retry..." && sleep 10 && pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt)

# Install service-specific dependencies (with retry logic)
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_api/requirements.txt || \
    (echo "First attempt failed, retrying..." && sleep 5 && pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_api/requirements.txt) || \
    (echo "Second attempt failed, final retry..." && sleep 10 && pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_api/requirements.txt)

# Stage 3: Application code
FROM dependencies AS application

# Copy shared libraries
COPY shared_libs/ /app/shared_libs/

# Copy service API source code
COPY services/service_api/src/ /app/services/service_api/src/

# Copy service_training source code (needed for schema imports)
COPY services/service_training/src/ /app/services/service_training/src/

# Copy necessary root configuration files
COPY config.toml /app/config.toml

# Set PYTHONPATH globally for all users and entrypoints
# Suppress gRPC ALTS warnings in Cloud Run
ENV PYTHONPATH=/app/shared_libs:/app/services/service_api/src:/app/services/service_training/src \
    CONFIG_TOML_PATH=/app/config.toml \
    GRPC_VERBOSITY=ERROR \
    GRPC_TRACE="" \
    GRPC_DNS_RESOLVER=native \
    GRPC_GOOGLE_CREDENTIALS=compute_engine \
    GRPC_EXPERIMENTAL_DISABLE_ALTS=1

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