# Select Python minor version used for both builder and runtime stages.
# Keep builder/runtime aligned to avoid wheel/Python ABI incompatibilities.
ARG PYTHON_VERSION=3.11

# Builder: compile wheels and build the project
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and sources
COPY pyproject.toml /app/

# Copy source tree
COPY src /app/src
COPY requirements-base.txt /app/requirements-base.txt

# Choose which requirements file to package (default: requirements-gpu.txt)
ARG REQS=requirements-gpu.txt
COPY ${REQS} /app/requirements.txt

# Build the project wheel only. Don't wheel the entire requirements.txt here
# (wheeling top-level requirements pulls many heavy packages and slows the
# builder). The runtime stage will install application dependencies; the
# project's wheel is the important artifact built here.
#
# NOTE: The following RUN uses BuildKit cache mounts to persist pip's cache
# between builds (speeds up repeated builds). This requires BuildKit to be
# enabled locally (DOCKER_BUILDKIT=1) or an equivalent cache mechanism in
# your CI. Cloud Build users should use the `cloudbuild.yaml` with Kaniko
# caching (provided in the repo) to get similar caching behavior in CI.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip build wheel setuptools \
    && python -m build -w -o /wheels /app \
    && cp /app/requirements*.txt /wheels/ || true


# Runtime stage: use the same Python minor as the builder to ensure wheel
# compatibility (set PYTHON_VERSION at build time if you need a different
# minor, e.g. --build-arg PYTHON_VERSION=3.12).
FROM python:${PYTHON_VERSION}-slim AS runtime

# Keep container output unbuffered and avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH=/app/src

WORKDIR /app

# Install minimal runtime deps. Installing Google Cloud SDK is optional
# (set --build-arg INSTALL_GCLOUD=true to include it).
ARG INSTALL_GCLOUD=false

# Allow passing the Google Cloud project at build or run time. You can
# override this at runtime with `docker run -e GOOGLE_CLOUD_PROJECT=...`.
ARG GOOGLE_CLOUD_PROJECT="laxai-466119"
ENV GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && if [ "${INSTALL_GCLOUD}" = "true" ]; then \
        apt-get install -y --no-install-recommends gnupg && \
        curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
        apt-get update && apt-get install -y --no-install-recommends google-cloud-sdk && \
        apt-get purge -y --auto-remove gnupg; \
    fi && rm -rf /var/lib/apt/lists/*

# Copy built wheels from the builder and install them from local wheel cache
COPY --from=builder /wheels /tmp/wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.8.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && if [ -f /tmp/wheels/requirements.txt ]; then python -m pip install --no-cache-dir -r /tmp/wheels/requirements.txt; fi \
    && python -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# Copy docs and config files explicitly so they are available in the image.
# NOTE: we intentionally do NOT copy source into the runtime image — the app
# code is provided by the installed wheel(s) produced in the builder stage.
COPY documentation /app/documentation
## WARNING: config.toml may contain secrets (API keys, tokens).
## Do NOT bake secrets into images. Prefer Cloud Build / Secret Manager
## injection at build or runtime. If config.toml is non-sensitive config
## (no secrets), it is safe to copy — otherwise remove this COPY and
## fetch the file from Secret Manager or mount it at runtime.
COPY config.toml /app/
# Also copy config.toml into the Python lib directory that the app
# may look for (example: /usr/local/lib/python3.12/config.toml). We
# compute the correct minor version at build time so images built with
# different Python minors will still work.
RUN PYDIR=$(python -c 'import sys; print(f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}")') \
    && mkdir -p "$PYDIR" \
    && cp /app/config.toml "$PYDIR/config.toml" || true
COPY src/config/gcs_structure.yaml /app/src/config/gcs_structure.yaml
# Do NOT copy files into /opt/conda (avoid coupling to conda paths / copying conda)

# Provide a lightweight runtime entrypoint module so `uvicorn main:app`
# works even though source is not copied into the image. This copies the
# top-level `src/main.py` into /app as `main.py`.
COPY src/main.py /app/main.py

# Create a non-root user for runtime and give them ownership of /app
RUN useradd -m -u 1000 laxai \
 && chown -R laxai:laxai /app

# Switch to non-root user for running the service
USER laxai
ENV HOME=/home/laxai

# Allow the runtime port to be configured at build or run time. Default is 8000.
ARG PORT=8000
ENV PORT=${PORT}

EXPOSE ${PORT}

# Default command: run uvicorn. Use shell form so ${PORT} is expanded at container start.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
