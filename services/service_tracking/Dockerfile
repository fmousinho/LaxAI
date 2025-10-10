# Multi-stage Dockerfile for service_tracking
# Stage 1: Base image with Python, PyTorch, and system dependencies
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS base

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
COPY services/service_tracking/requirements.txt /app/services/service_tracking/requirements.txt

# Install shared libraries dependencies first (with retry logic)
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt || \
    (echo "First attempt failed, retrying..." && sleep 5 && pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt) || \
    (echo "Second attempt failed, final retry..." && sleep 10 && pip install --no-cache-dir --timeout=300 --retries=10 -r shared_libs/requirements.txt)

# Install service-specific dependencies (with retry logic)
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_tracking/requirements.txt || \
    (echo "First attempt failed, retrying..." && sleep 5 && pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_tracking/requirements.txt) || \
    (echo "Second attempt failed, final retry..." && sleep 10 && pip install --no-cache-dir --timeout=300 --retries=10 -r services/service_tracking/requirements.txt)

# Stage 3: Application code
FROM dependencies AS application

# Copy shared libraries
COPY shared_libs/ /app/shared_libs/

# Copy service tracking source code
COPY services/service_tracking/src/ /app/services/service_tracking/src/

# Copy necessary root configuration files
COPY config.toml /app/config.toml

# Set PYTHONPATH globally for all users and entrypoints
# Suppress gRPC ALTS warnings in Cloud Run
ENV PYTHONPATH=/app/shared_libs:/app/services/service_tracking/src \
    CONFIG_TOML_PATH=/app/config.toml \
    GRPC_VERBOSITY=ERROR \
    GRPC_TRACE=""

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create matplotlib config directory and give appuser permission
RUN mkdir -p /home/appuser/.config/matplotlib && chown -R appuser:appuser /home/appuser

# Create fontconfig cache directory and give appuser permission
RUN mkdir -p /home/appuser/.cache/fontconfig && chown -R appuser:appuser /home/appuser/.cache

RUN chown -R appuser:appuser /app
USER appuser

# Working directory for the service
WORKDIR /app/

# Default command - can be overridden
CMD ["sh", "-c", "python services/service_tracking/src/main.py --tenant_id=default"]

# Expose port if running as web service
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Stage 4: Testing (optional, can be used for CI/CD)
# FROM application AS testing

# # Install additional testing dependencies
# RUN pip install --no-cache-dir pytest-cov pytest-xdist

# # Run tests (optional, uncomment if you want tests to run during build)
# # RUN python -m pytest tests/ -v

# Final production stage
FROM application AS production

# Optimize Python
ENV PYTHONOPTIMIZE=1
