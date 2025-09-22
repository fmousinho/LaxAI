#!/bin/bash

############################################################################
#                                                                          #
#               PUB/SUB INFRASTRUCTURE SETUP FOR TRACKING PROXY           #
#                                                                          #
############################################################################

# This script sets up the Pub/Sub infrastructure needed for the tracking proxy
# Run this script with appropriate GCP permissions before deploying the Cloud Function

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-laxai-466119}"
TOPIC_NAME="tracking-jobs"
REGION="us-central1"
SERVICE_ACCOUNT="laxai-466119@appspot.gserviceaccount.com"

echo "Setting up Pub/Sub infrastructure for tracking proxy..."
echo "Project: $PROJECT_ID"
echo "Topic: $TOPIC_NAME"
echo "Region: $REGION"
echo "Service Account: $SERVICE_ACCOUNT"
echo

# Create the Pub/Sub topic if it doesn't exist
echo "Creating Pub/Sub topic: $TOPIC_NAME"
if gcloud pubsub topics describe "$TOPIC_NAME" --project="$PROJECT_ID" &>/dev/null; then
    echo "Topic $TOPIC_NAME already exists"
else
    gcloud pubsub topics create "$TOPIC_NAME" \
        --project="$PROJECT_ID" \
        --labels="service=tracking-proxy"
    echo "Created topic: $TOPIC_NAME"
fi

echo

# Grant IAM permissions to the service account
echo "Granting IAM permissions to service account..."

# Cloud Run Developer role
echo "Granting Cloud Run Developer role..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/run.developer" \
    --condition=None \
    --quiet

# Cloud Datastore User role
echo "Granting Cloud Datastore User role..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/datastore.user" \
    --condition=None \
    --quiet

# Pub/Sub Subscriber role
echo "Granting Pub/Sub Subscriber role..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/pubsub.subscriber" \
    --condition=None \
    --quiet

echo
echo "Pub/Sub infrastructure setup complete!"
echo
echo "Next steps:"
echo "1. Deploy the tracking proxy Cloud Function using:"
echo "   gcloud builds submit . --config=deploy/cloudbuild/cloudbuild-trackingproxy-functions.yaml"
echo
echo "2. Test the setup by publishing a message to the topic:"
echo "   gcloud pubsub topics publish $TOPIC_NAME --message='{\"action\":\"create\",\"tenant_id\":\"test\",\"custom_name\":\"test-job\"}' --project=$PROJECT_ID"