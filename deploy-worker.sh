#!/bin/bash

# Cloud Run Job deployment script for LaxAI training workers
# This script sets up the complete infrastructure for GPU training jobs

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-}"
REGION="${REGION:-us-central1}"
SERVICE_ACCOUNT_NAME="googlebatch"  # Use existing service account
SERVICE_ACCOUNT_EMAIL="googlebatch@${PROJECT_ID}.iam.gserviceaccount.com"
PUBSUB_TOPIC="training-jobs"
SUBSCRIPTION_NAME="training-jobs-sub"
ARTIFACT_REGISTRY_REPO="laxai-repo"
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/laxai-worker"
WORKER_SERVICE_NAME="laxai-worker"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if PROJECT_ID is set
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "GOOGLE_CLOUD_PROJECT environment variable is not set."
        log_info "Please run: export GOOGLE_CLOUD_PROJECT=your-project-id"
        exit 1
    fi
    
    log_info "Prerequisites check passed ✓"
}

enable_apis() {
    log_info "Enabling required Google Cloud APIs..."
    
    gcloud services enable run.googleapis.com \
        pubsub.googleapis.com \
        firestore.googleapis.com \
        cloudbuild.googleapis.com \
        containerregistry.googleapis.com \
        --project=$PROJECT_ID
    
    log_info "APIs enabled ✓"
}

create_service_account() {
    log_info "Using existing service account: $SERVICE_ACCOUNT_EMAIL"
    
    # Check if service account exists
    if gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL --project=$PROJECT_ID &>/dev/null; then
        log_info "Service account exists ✓"
        
        # Grant additional permissions needed for training jobs
        ADDITIONAL_PERMISSIONS=(
            "pubsub.subscriber"
        )
        
        for permission in "${ADDITIONAL_PERMISSIONS[@]}"; do
            gcloud projects add-iam-policy-binding $PROJECT_ID \
                --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
                --role="roles/${permission}" || log_warn "Permission may already be granted"
        done
        
        log_info "Additional permissions granted ✓"
    else
        log_error "Service account $SERVICE_ACCOUNT_EMAIL does not exist!"
        log_info "Please run your existing Cloud Build setup first or create the service account"
        exit 1
    fi
}

setup_pubsub() {
    log_info "Setting up Pub/Sub topic and subscription..."
    
    # Create topic
    gcloud pubsub topics create $PUBSUB_TOPIC --project=$PROJECT_ID || log_warn "Topic may already exist"
    
    # Create subscription
    gcloud pubsub subscriptions create $SUBSCRIPTION_NAME \
        --topic=$PUBSUB_TOPIC \
        --ack-deadline=600 \
        --message-retention-duration=7d \
        --project=$PROJECT_ID || log_warn "Subscription may already exist"
    
    log_info "Pub/Sub setup complete ✓"
}

build_and_push_image() {
    log_info "Building and pushing Docker image to Artifact Registry..."
    
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
    
    # Build the image using the same repo structure as your existing setup
    docker build -f src/cloud/Dockerfile.worker -t $IMAGE_NAME:latest .
    
    # Push to Artifact Registry
    docker push $IMAGE_NAME:latest
    
    log_info "Docker image built and pushed to Artifact Registry ✓"
}

generate_job_config() {
    log_info "Generating Cloud Run Job configuration..."
    
    cat > cloud-run-job.yaml << EOF
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: $WORKER_SERVICE_NAME
  labels:
    cloud.googleapis.com/location: $REGION
spec:
  template:
    spec:
      template:
        spec:
          restartPolicy: Never
          taskTimeoutSeconds: 25200  # 7 hours
          serviceAccountName: $SERVICE_ACCOUNT_EMAIL
          containers:
          - name: training-worker
            image: $IMAGE_NAME:latest
            env:
            - name: GOOGLE_CLOUD_PROJECT
              value: "$PROJECT_ID"
            - name: TRAINING_JOBS_SUBSCRIPTION
              value: "$SUBSCRIPTION_NAME"
            - name: FIRESTORE_ENABLED
              value: "true"
            - name: WORKER_TIMEOUT
              value: "25200"
            - name: MAX_CONCURRENT_JOBS
              value: "1"
            resources:
              limits:
                cpu: "4000m"
                memory: "16Gi"
                nvidia.com/gpu: "1"
              requests:
                cpu: "2000m"
                memory: "8Gi"
                nvidia.com/gpu: "1"
          nodeSelector:
            run.googleapis.com/accelerator: nvidia-l4
EOF
    
    log_info "Job configuration generated ✓"
}

deploy_job() {
    log_info "Deploying Cloud Run Job..."
    
    gcloud run jobs replace cloud-run-job.yaml \
        --region=$REGION \
        --project=$PROJECT_ID
    
    log_info "Cloud Run Job deployed ✓"
}

show_usage() {
    log_info "Deployment complete! Here's how to use the system:"
    echo ""
    echo "1. Submit training jobs via Cloud Function:"
    echo "   curl -X POST https://your-function-url/orchestrate \\"
    echo "        -H 'Content-Type: application/json' \\"
    echo "        -d '{\"tenant_id\": \"test\", \"training_params\": {...}}'"
    echo ""
    echo "2. Check job status:"
    echo "   curl 'https://your-function-url/status?task_id=<task_id>'"
    echo ""
    echo "3. Execute worker manually (for testing):"
    echo "   gcloud run jobs execute laxai-worker --region=$REGION"
    echo ""
    echo "4. View logs:"
    echo "   gcloud logging read 'resource.type=\"cloud_run_job\"' --limit=50"
    echo ""
    log_info "Setup complete! 🚀"
}

# Main execution
main() {
    log_info "Starting LaxAI Cloud Run Job deployment..."
    
    check_prerequisites
    enable_apis
    create_service_account
    setup_pubsub
    build_and_push_image
    generate_job_config
    deploy_job
    show_usage
}

# Handle command line arguments
case "${1:-deploy}" in
    "check")
        check_prerequisites
        ;;
    "build")
        check_prerequisites
        build_and_push_image
        ;;
    "deploy")
        main
        ;;
    "clean")
        log_info "Cleaning up resources..."
        gcloud run jobs delete $WORKER_SERVICE_NAME --region=$REGION --quiet || true
        gcloud pubsub subscriptions delete $SUBSCRIPTION_NAME --quiet || true
        gcloud pubsub topics delete $PUBSUB_TOPIC --quiet || true
        log_info "Cleanup complete"
        ;;
    *)
        echo "Usage: $0 {deploy|build|check|clean}"
        echo "  deploy - Full deployment (default)"
        echo "  build  - Build and push image only"
        echo "  check  - Check prerequisites only"
        echo "  clean  - Clean up all resources"
        exit 1
        ;;
esac
