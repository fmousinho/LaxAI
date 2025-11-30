ARG BASE_IMAGE=fmousinho/laxai-gpu-base:latest
FROM ${BASE_IMAGE}

# Environment Settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONOPTIMIZE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

ARG GIT_COMMIT_SHA=unknown
ENV GIT_COMMIT_SHA=${GIT_COMMIT_SHA}

WORKDIR /app

# Copy code with correct ownership
COPY --chown=appuser:appuser shared_libs/ shared_libs/
COPY --chown=appuser:appuser services/service_training/src/ services/service_training/src/
COPY --chown=appuser:appuser config.toml config.toml

USER appuser

# Python path
ENV PYTHONPATH=/app:/app/services/service_training/src

# Create matplotlib and fontconfig dirs
RUN mkdir -p ~/.config/matplotlib ~/.cache/fontconfig

# Default command
CMD ["python", "services/service_training/src/main.py"]

# Expose port
EXPOSE 8000

# HEALTHCHECK (only if curl exists)
# If curl isn't in the base image, replace with python-based check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1
