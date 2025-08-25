# Allow selecting the runtime base image (used by the runtime stage below)
# Default to a slim Python base for small inference images
ARG BASE_IMAGE=python:3.12-slim
# A dependency-only image that provides prebuilt wheels in /wheels.
# Default tag is the latest known deps-image built locally; override with
# --build-arg DEPS_IMAGE=... when building the app image.
ARG DEPS_IMAGE=fmousinho/laxai-deps:latest

# Builder: compile wheels and build the project
FROM ${BASE_IMAGE} AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # build-essential \
    # gcc \
    # curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and sources
COPY pyproject.toml /app/

# Copy source tree
COPY src /app/src


# Build the project wheel only. Don't wheel the entire requirements.txt here
# (wheeling top-level requirements pulls many heavy packages and slows the
# builder). The runtime stage will install application dependencies; the
# project's wheel is the important artifact built here.
# Build the project wheel only. Don't wheel the entire requirements here.
RUN python -m pip install --upgrade pip build wheel setuptools \
    && python -m build -w -o /wheels /app

# Runtime stage: base the runtime directly on the dependency-only image so
# the runtime inherits the preinstalled binary deps (torch, etc.). This
# avoids copying requirement files or wheels from the deps image which may
# not expose /wheels.
FROM ${DEPS_IMAGE} AS runtime

# Keep container output unbuffered and avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH=/app/src

WORKDIR /app


# Allow passing the Google Cloud project at build or run time. You can
# override this at runtime with `docker run -e GOOGLE_CLOUD_PROJECT=...`.
ARG GOOGLE_CLOUD_PROJECT="laxai-466119"
ENV GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}



# Copy dependency wheels from the deps image and then overlay the app wheel
# produced by the builder. The deps image is expected to include all
# dependency wheels (binary packages like torch) in /wheels.
# copy app wheel(s) from builder into a temp directory and install them into
# the runtime environment. The deps image is expected to already provide the
# heavy binary deps (torch) inside its venv (for example /opt/venv). We try
# to use /opt/venv/bin/pip if present, otherwise fall back to python -m pip.
COPY --from=builder /wheels/ /tmp/wheels/

RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl; \
    rm -rf /tmp/wheels

# Copy docs and config files explicitly so they are available in the image.
# NOTE: we intentionally do NOT copy source into the runtime image — the app
# code is provided by the installed wheel(s) produced in the builder stage.
COPY documentation /app/documentation
COPY config.toml /app/
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
