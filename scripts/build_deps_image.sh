#!/usr/bin/env bash
# Build helper for the dependency-only Docker image.
# Usage: scripts/build_deps_image.sh [--push] [--registry REGISTRY] [--project PROJECT_ID]
# Example: scripts/build_deps_image.sh --push --registry gcr.io --project my-gcp-project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PUSH=false
REGISTRY="docker.io"
PROJECT_ID="fmousinho"
IMAGE_NAME="laxai"
DOCKERFILE="Dockerfile.deps"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --push) PUSH=true; _EXPLICIT_PUSH=true; shift ;;
    --registry) REGISTRY="$2"; shift 2 ;;
  --multi-arch) MULTIARCH=true; shift ;;
    --project) PROJECT_ID="$2"; shift 2 ;;
    --image) IMAGE_NAME="$2"; shift 2 ;;
    --dockerfile) DOCKERFILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Allow forcing auto-push via environment variable (AUTO_PUSH=true)
if [[ -n "${AUTO_PUSH:-}" ]]; then
  if [[ "$(echo "$AUTO_PUSH" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    PUSH=true
  fi
fi

# If running in CI and a registry is provided, enable auto-push to simplify CI pipelines
if [[ "$PUSH" != true && -n "$REGISTRY" ]]; then
  if [[ -n "${CI:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${GITLAB_CI:-}" || -n "${BUILDKITE:-}" ]]; then
    echo "CI environment detected and registry provided — enabling auto-push"
    PUSH=true
  fi
fi
# Find all requirements-like files to include in the hash (sorted)
# Use nullglob so patterns that don't match expand to nothing
shopt -s nullglob
REQ_FILES=(requirements*.txt requirements*.pip requirements*.in)
shopt -u nullglob

EXISTS=("")
EXISTS=()
for f in "${REQ_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    EXISTS+=("$f")
  fi
done

if [[ ${#EXISTS[@]} -eq 0 ]]; then
  echo "No requirements files found in repo root. Ensure requirements.txt exists."
  exit 1
fi

 # Compute a stable short hash based on contents and filenames (order-insensitive)
if [[ ${#EXISTS[@]} -eq 0 ]]; then
  echo "No requirements files found in repo root. Ensure requirements.txt exists."
  exit 1
fi

# Compute short sha1 hash from filenames and their contents (stable order)
TMP_HASH_FILE=$(mktemp)
for fn in "${EXISTS[@]}"; do
  echo "$fn" >> "$TMP_HASH_FILE"
  cat "$fn" >> "$TMP_HASH_FILE"
done
HASH=$(sha1sum "$TMP_HASH_FILE" | awk '{print substr($1,1,12)}')
rm -f "$TMP_HASH_FILE"

# Compose image reference
if [[ -n "$REGISTRY" ]]; then
  if [[ -n "$PROJECT_ID" ]]; then
    IMAGE_REF="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:${HASH}"
  else
    IMAGE_REF="${REGISTRY}/${IMAGE_NAME}:${HASH}"
  fi
else
  # default to local docker name
  IMAGE_REF="${IMAGE_NAME}:${HASH}"
fi

echo "Requirements files considered:"
for f in "${EXISTS[@]}"; do echo "  - $f"; done

echo "Computed hash: $HASH"
echo "Building dependency image: $IMAGE_REF"

# Create a temporary build context that contains the Dockerfile and a requirements/ dir
TMP_DIR=$(mktemp -d)
mkdir -p "$TMP_DIR/requirements"
cp "$DOCKERFILE" "$TMP_DIR/Dockerfile"
for f in "${EXISTS[@]}"; do
  cp "$f" "$TMP_DIR/requirements/$(basename "$f")"
done


if [[ "${MULTIARCH:-}" == "true" ]]; then
  echo "Multi-arch build requested. Using docker buildx."
  # Ensure a builder exists and is bootstrapped
  docker buildx inspect multi-builder >/dev/null 2>&1 || docker buildx create --name multi-builder --use
  docker buildx inspect --bootstrap

  # If pushing, use --push to publish manifest; otherwise load the image locally (may not support multi-arch)
  if [[ "$PUSH" == true ]]; then
    docker buildx build --platform linux/amd64,linux/arm64 -f "$TMP_DIR/Dockerfile" -t "$IMAGE_REF" --push "$TMP_DIR"
  else
    # Try to build and load the default platform (builder will select local platform)
    docker buildx build --platform linux/amd64,linux/arm64 -f "$TMP_DIR/Dockerfile" -t "$IMAGE_REF" --load "$TMP_DIR"
  fi
else
  docker build -f "$TMP_DIR/Dockerfile" -t "$IMAGE_REF" "$TMP_DIR"
fi

# Clean up temporary dir
rm -rf "$TMP_DIR"

if [[ "$PUSH" == true ]]; then
  echo "Pushing $IMAGE_REF"
  docker push "$IMAGE_REF"
fi

if [[ "$PUSH" == true && -t 1 && -z "${CI:-}" && -z "${GITHUB_ACTIONS:-}" ]]; then
  # If we will push but the user didn't explicitly pass --push, confirm interactively
  if [[ -z "${_EXPLICIT_PUSH:-}" ]]; then
    read -p "About to push $IMAGE_REF to registry. Continue? [y/N] " yn
    case "$yn" in
      [Yy]*) echo "Proceeding with push..." ;;
      *) echo "Push cancelled by user."; exit 0 ;;
    esac
  fi
fi

echo "Done. Use this image as the base for fast app builds."
