# Multi-stage Dockerfile for service_cr_train_proxy
# Stage 1: Base image with Python and system dependencies
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy shared libraries requirements first for better caching
COPY shared_libs/requirements.txt /app/shared_libs/requirements.txt

# Copy service-specific requirements
COPY services/service_cr_train_proxy/requirements.txt /app/services/service_cr_train_proxy/requirements.txt

# Install shared libraries dependencies first
RUN pip install --no-cache-dir -r shared_libs/requirements.txt

# Install service-specific dependencies
RUN pip install --no-cache-dir -r services/service_cr_train_proxy/requirements.txt

# Stage 3: Application code
FROM dependencies AS application

# Copy shared libraries
COPY shared_libs/ /app/shared_libs/

# Copy service source code
COPY services/service_cr_train_proxy/src/ /app/services/service_cr_train_proxy/src/

# Copy necessary root configuration files
COPY config.toml /app/config.toml

# Set PYTHONPATH globally for all users and entrypoints
ENV PYTHONPATH=/app/shared_libs:/app/services/service_cr_train_proxy/src

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Working directory for the service
WORKDIR /app/

# Default command - can be overridden
CMD ["sh", "-c", "python services/service_cr_train_proxy/src/main.py"]

# Final production stage
FROM application AS production

# Remove development dependencies and cleanup
RUN pip uninstall -y pytest black flake8 || true

# Optimize Python
ENV PYTHONOPTIMIZE=1
