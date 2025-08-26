#!/bin/bash

# Simple deployment script that uses shared configuration
# Usage: ./scripts/deploy.sh [main|worker|both]

set -e

# Load shared configuration
source ./src/config/cloud-config.env

# Check required variables
if [[ -z "$GOOGLE_CLOUD_PROJECT" ]]; then
    echo "Error: GOOGLE_CLOUD_PROJECT not set"
    echo "Run: export GOOGLE_CLOUD_PROJECT=your-project-id"
    exit 1
fi

deploy_main() {
    echo "Deploying main application..."
    # Use the repository root as the build context so Dockerfiles that COPY
    # top-level paths (like requirements/) work regardless of the current cwd.
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    gcloud builds submit --config deploy/cloudbuild/cloudbuild.yaml \
        --substitutions=_DEPLOY_TO_CLOUD_RUN=true,_REQS=${CLOUD_REQS},_CPU_LIMIT=${MAIN_CPU_LIMIT},_MEMORY_LIMIT=${MAIN_MEMORY_LIMIT} \
        "$REPO_ROOT"
}

deploy_worker() {
    echo "Deploying training worker..."
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    gcloud builds submit --config deploy/cloudbuild/cloudbuild-worker.yaml \
        --substitutions=_DEPLOY_TO_CLOUD_RUN=true,_REQS=${GPU_REQS} \
        "$REPO_ROOT"
}

case "${1:-both}" in
    "main")
        deploy_main
        ;;
    "worker")
        deploy_worker
        ;;
    "both")
        deploy_main
        deploy_worker
        ;;
    *)
        echo "Usage: $0 [main|worker|both]"
        exit 1
        ;;
esac

echo "✓ Deployment complete!"
echo "Configuration used:"
echo "  Region: $CLOUD_REGION"
echo "  GPU: $GPU_COUNT x $GPU_TYPE"
echo "  Resources: $MAIN_CPU_LIMIT CPU, $MAIN_MEMORY_LIMIT memory"
