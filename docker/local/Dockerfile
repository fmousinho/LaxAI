# Allow selecting the runtime base image (used by the runtime stage below)
# We will use a more robust Python base image since we need to install dependencies
ARG BASE_IMAGE=python:3.12-slim
ARG REQS=requirements/requirements-cloud.txt

# Builder: compile wheels and build the project
FROM ${BASE_IMAGE} AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and sources
COPY pyproject.toml /app/

# Copy source tree
COPY src /app/src

# Build the project wheel only.
RUN python -m pip install --upgrade pip build wheel setuptools \
    && python -m build -w -o /wheels /app


# Runtime stage: This is the modified stage.
# It now installs all dependencies from requirements directly.
FROM ${BASE_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PYTHONPATH=/app/src

WORKDIR /app

# Copy the requirements file and install dependencies
# Note: You'll need to create a requirements.txt file at your project root
COPY ${REQS} /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application wheel and install
COPY --from=builder /wheels/ /tmp/wheels/
RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl; \
    rm -rf /tmp/wheels

# Copy docs and config files
COPY documentation /app/documentation
COPY config.toml /app/
COPY src/config/gcs_structure.yaml /app/src/config/gcs_structure.yaml

# Provide a lightweight runtime entrypoint module
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