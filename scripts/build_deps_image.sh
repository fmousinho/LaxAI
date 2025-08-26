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
IMAGE_NAME="laxai-deps"
DOCKERFILE="docker/base/Dockerfile.deps"
MULTIARCH=false
# Tagging behaviour: hash (default), timestamp, git, or explicit
TAG_MODE="hash"
EXPLICIT_TAG=""
# Optional: remove other images with same repo/name after successful build
PRUNE_OLD=false
NO_CACHE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
  --push) PUSH=true; _EXPLICIT_PUSH=true; shift ;;
    --registry) REGISTRY="$2"; shift 2 ;;
  --multi-arch) MULTIARCH=true; shift ;;
    --project) PROJECT_ID="$2"; shift 2 ;;
    --tag-mode) TAG_MODE="$2"; shift 2 ;;
    --tag) EXPLICIT_TAG="$2"; shift 2 ;;
    --prune-old) PRUNE_OLD=true; shift ;;
  --no-cache) NO_CACHE=true; shift ;;
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
# Prefer requirements inside the `requirements/` directory (recommended).
# This keeps the deps image focused on the repo's declared files.
REQ_DIR="requirements"
EXISTS=()
PREFERRED_REQS=("$REQ_DIR/requirements-base.txt" "$REQ_DIR/requirements-cpu.txt")
for f in "${PREFERRED_REQS[@]}"; do
  if [[ -f "$f" ]]; then
    EXISTS+=("$f")
  fi
done

if [[ ${#EXISTS[@]} -eq 0 ]]; then
  # Fallback: look for any requirements in requirements/ first, then repo root
  shopt -s nullglob
  ALL_REQS=("$REQ_DIR"/requirements*.txt "$REQ_DIR"/requirements*.pip "$REQ_DIR"/requirements*.in requirements*.txt requirements*.pip requirements*.in)
  shopt -u nullglob
  for f in "${ALL_REQS[@]:-}"; do
    if [[ -f "$f" ]]; then
      EXISTS+=("$f")
    fi
  done
fi

if [[ ${#EXISTS[@]} -eq 0 ]]; then
  echo "No requirements files found in repo (checked $REQ_DIR and repo root). Ensure requirements files exist."
  exit 1
fi

# Compute short sha1 hash from filenames and their contents (stable order)
TMP_HASH_FILE=$(mktemp)
for fn in "${EXISTS[@]}"; do
  # write basename first to keep order stable and avoid including full paths
  echo "$(basename "$fn")" >> "$TMP_HASH_FILE"
  cat "$fn" >> "$TMP_HASH_FILE"
done
# Use a portable sha1 command (macOS has shasum)
if command -v sha1sum >/dev/null 2>&1; then
  HASH=$(sha1sum "$TMP_HASH_FILE" | awk '{print substr($1,1,12)}')
else
  HASH=$(shasum -a 1 "$TMP_HASH_FILE" | awk '{print substr($1,1,12)}')
fi
rm -f "$TMP_HASH_FILE"

# Compose image reference

# Determine tag based on mode
if [[ -n "$EXPLICIT_TAG" ]]; then
  TAG="$EXPLICIT_TAG"
else
  case "$TAG_MODE" in
    hash)
      TAG="$HASH"
      ;;
    timestamp)
      TAG="$(date -u +%Y%m%dT%H%M%SZ)"
      ;;
    git)
      if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        TAG="$(git rev-parse --short=12 HEAD)"
      else
        echo "git tag mode requested but git not available; falling back to hash"
        TAG="$HASH"
      fi
      ;;
    *)
      echo "Unknown TAG_MODE: $TAG_MODE. Falling back to hash.";
      TAG="$HASH";
      ;;
  esac
fi

# Build fully-qualified image reference
if [[ -n "$REGISTRY" ]]; then
  if [[ -n "$PROJECT_ID" ]]; then
    IMAGE_REF="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
  else
    IMAGE_REF="${REGISTRY}/${IMAGE_NAME}:${TAG}"
  fi
else
  IMAGE_REF="${IMAGE_NAME}:${TAG}"
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
  basefn=$(basename "$f")
  cp "$f" "$TMP_DIR/requirements/$basefn"
  # Also copy into the build context root to support Dockerfiles that expect
  # requirements files at the context root (some Dockerfiles use -r requirements-base.txt)
  cp "$f" "$TMP_DIR/$basefn"
done


if [[ "${MULTIARCH:-}" == "true" ]]; then
  echo "Multi-arch build requested. Using docker buildx."
  # Ensure a builder exists and is bootstrapped
  docker buildx inspect multi-builder >/dev/null 2>&1 || docker buildx create --name multi-builder --use
  docker buildx inspect --bootstrap

  # If pushing, use --push to publish manifest; otherwise load the image locally (may not support multi-arch)
  if [[ "$PUSH" == true ]]; then
    # Push a multi-arch manifest (amd64 + arm64)
  DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64,linux/arm64 -f "$TMP_DIR/Dockerfile" -t "$IMAGE_REF" ${NO_CACHE:+--no-cache} --push "$TMP_DIR"

    # After pushing a multi-arch manifest, also create/update the :latest tag in the registry
    # so consumers that pull ':latest' see the same multi-arch index. Use buildx imagetools
    # which creates a manifest list referencing the pushed images.
    if command -v docker >/dev/null 2>&1 && docker buildx imagetools --help >/dev/null 2>&1; then
      if [[ -n "$REGISTRY" && -n "$PROJECT_ID" ]]; then
        CREATE_LATEST_REF="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:latest"
      elif [[ -n "$REGISTRY" ]]; then
        CREATE_LATEST_REF="${REGISTRY}/${IMAGE_NAME}:latest"
      else
        CREATE_LATEST_REF="${IMAGE_NAME}:latest"
      fi
      echo "Creating multi-arch manifest tag ${CREATE_LATEST_REF} -> ${IMAGE_REF}"
      # Try to create the remote :latest multi-arch manifest. If successful,
      # set IMAGETOOLS_CREATED so we don't later push a local single-arch :latest
      # which would overwrite the multi-arch manifest.
      if docker buildx imagetools create --tag "$CREATE_LATEST_REF" "$IMAGE_REF"; then
        IMAGETOOLS_CREATED=true
      else
        IMAGETOOLS_CREATED=false
      fi
    else
      echo "docker buildx imagetools not available; skipping :latest manifest creation"
    fi
  else
    # When not pushing, --load only supports a single platform. Pick the local platform
    # so the built image can be loaded into the local docker daemon.
    UNAME_M="$(uname -m)"
    case "$UNAME_M" in
      x86_64|amd64) LOCAL_PLATFORM="linux/amd64" ;;
      arm64|aarch64) LOCAL_PLATFORM="linux/arm64" ;;
      *) LOCAL_PLATFORM="linux/amd64" ;;
    esac
    echo "Building for local platform $LOCAL_PLATFORM (use --push to publish multi-arch)"
  DOCKER_BUILDKIT=1 docker buildx build --platform "$LOCAL_PLATFORM" -f "$TMP_DIR/Dockerfile" -t "$IMAGE_REF" ${NO_CACHE:+--no-cache} --load "$TMP_DIR"
  fi
else
  DOCKER_BUILDKIT=1 docker build -f "$TMP_DIR/Dockerfile" -t "$IMAGE_REF" ${NO_CACHE:+--no-cache} "$TMP_DIR"
fi

# Clean up temporary dir
rm -rf "$TMP_DIR"

if [[ "$PUSH" == true ]]; then
  echo "Pushing $IMAGE_REF"
  docker push "$IMAGE_REF"
fi

# Tag the newly built image also as :latest (local tag)
if [[ -n "$REGISTRY" && -n "$PROJECT_ID" ]]; then
  LATEST_REF="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:latest"
elif [[ -n "$REGISTRY" ]]; then
  LATEST_REF="${REGISTRY}/${IMAGE_NAME}:latest"
else
  LATEST_REF="${IMAGE_NAME}:latest"
fi
echo "Tagging ${IMAGE_REF} -> ${LATEST_REF}"
docker tag "$IMAGE_REF" "$LATEST_REF"

# If we're pushing, push the :latest tag as well. However, if we already
# created a multi-arch :latest manifest via buildx imagetools above, skip the
# local push to avoid overwriting it with a single-arch reference.
if [[ "$PUSH" == true ]]; then
  if [[ "${MULTIARCH:-}" == "true" && "${IMAGETOOLS_CREATED:-}" == "true" ]]; then
    echo "Multi-arch :latest manifest already created remotely; skipping push of ${LATEST_REF}"
  else
    echo "Pushing ${LATEST_REF}"
    docker push "$LATEST_REF"
  fi
fi

# Unconditionally remove all other local images with the same repo/name
# Keep only the new tag ($TAG) and the 'latest' tag
if [[ -n "$REGISTRY" && -n "$PROJECT_ID" ]]; then
  PREFIX="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}"
elif [[ -n "$REGISTRY" ]]; then
  PREFIX="${REGISTRY}/${IMAGE_NAME}"
else
  PREFIX="${IMAGE_NAME}"
fi

# docker image ls typically shows images as 'project/image:tag' for docker.io, not
# 'docker.io/project/image:tag'. Build a LOCAL_PREFIX that matches docker image ls output
# so grep finds local image repository names correctly.
if [[ -z "$REGISTRY" || "$REGISTRY" == "docker.io" ]]; then
  if [[ -n "$PROJECT_ID" ]]; then
    LOCAL_PREFIX="${PROJECT_ID}/${IMAGE_NAME}"
  else
    LOCAL_PREFIX="${IMAGE_NAME}"
  fi
else
  LOCAL_PREFIX="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}"
fi

echo "Pruning (by image id) for ${LOCAL_PREFIX} — keeping tag :${TAG} and :latest"


# Resolve the newly built image id (normalize to docker image ls short id)
NEW_ID_FULL=$(docker image inspect -f '{{.Id}}' "$IMAGE_REF" 2>/dev/null || true)
# strip possible 'sha256:' prefix and take the first 12 chars to match `docker image ls` output
NEW_ID=$(echo "${NEW_ID_FULL:-}" | sed -e 's/^sha256://g' | cut -c1-12)
echo "New image id (short): ${NEW_ID:-<unknown>}"

# Gather all unique image IDs for this repo prefix
ALL_IDS=$(docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep "^${LOCAL_PREFIX}:" | awk '{print $2}' | sort -u || true)

if [[ -z "${ALL_IDS:-}" ]]; then
  echo "No local images found for prefix ${LOCAL_PREFIX}. Nothing to prune."
else
  # Build candidate list: all ids except NEW_ID
  CANDIDATES=()
  for id in ${ALL_IDS}; do
    if [[ -n "$NEW_ID" && "$id" == "$NEW_ID" ]]; then
      continue
    fi
    CANDIDATES+=("$id")
  done

  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "No old image IDs to remove (only the new image id is present)."
  else
    echo "Found the following old image IDs for ${LOCAL_PREFIX}:"
    for id in "${CANDIDATES[@]}"; do
      echo "  - $id"
    done

    # Unconditionally remove candidate image IDs (no interactive prompt).
    for id in "${CANDIDATES[@]}"; do
      echo "Removing image id $id"
      docker rmi -f "$id" || true
    done
  fi
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
