# Multi-stage Dockerfile for service_training
# Stage 1: Base image with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3-opencv \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy shared libraries requirements first for better caching
COPY shared_libs/requirements.txt /app/shared_libs/requirements.txt

# Copy service-specific requirements
COPY services/service_training/requirements.txt /app/services/service_training/requirements.txt

# Install shared libraries dependencies first
RUN pip install --no-cache-dir -r shared_libs/requirements.txt

# Install service-specific dependencies
RUN pip install --no-cache-dir -r services/service_training/requirements.txt

# Stage 3: Application code
FROM dependencies as application

# Copy shared libraries
COPY shared_libs/ /app/shared_libs/


# Copy service training source code
COPY services/service_training/ /app/services/service_training/

# Copy necessary root configuration files
COPY config.toml /app/config.toml
COPY pyrightconfig.json /app/pyrightconfig.json
COPY pytest.ini /app/pytest.ini

# Set PYTHONPATH globally for all users and entrypoints
ENV PYTHONPATH=/app/shared_libs:/app/services/service_training/src

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Working directory for the service
WORKDIR /app

# Default command - can be overridden
CMD ["python", "src/main.py"]

# Expose port if running as web service
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Stage 4: Testing (optional, can be used for CI/CD)
FROM application as testing

# Install additional testing dependencies
RUN pip install --no-cache-dir pytest-cov pytest-xdist

# Run tests (optional, uncomment if you want tests to run during build)
# RUN python -m pytest tests/ -v

# Final production stage
FROM application as production

# Remove development dependencies and cleanup
RUN pip uninstall -y pytest black flake8 || true

# Optimize Python
ENV PYTHONOPTIMIZE=1
