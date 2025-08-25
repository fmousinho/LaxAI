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
    gcloud builds submit --config deploy/cloudbuild/cloudbuild.yaml \
        --substitutions=_DEPLOY_TO_CLOUD_RUN=true,_REQS=${CLOUD_REQS},_CPU_LIMIT=${MAIN_CPU_LIMIT},_MEMORY_LIMIT=${MAIN_MEMORY_LIMIT}
}

deploy_worker() {
    echo "Deploying training worker..."
    gcloud builds submit --config deploy/cloudbuild/cloudbuild-worker.yaml \
        --substitutions=_DEPLOY_TO_CLOUD_RUN=true,_REQS=${GPU_REQS}
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
echo "  Resources: $CPU_LIMIT CPU, $MEMORY_LIMIT memory"
