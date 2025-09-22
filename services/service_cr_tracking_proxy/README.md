# LaxAI Tracking Proxy Service

This service provides a Pub/Sub-triggered Cloud Function proxy for executing tracking jobs on Google Cloud Run.

## Overview

The tracking proxy acts as a bridge between Pub/Sub messages and Cloud Run jobs for video tracking tasks. It follows the same architecture as the training proxy but handles tracking-specific workflows.

## Architecture

```
Pub/Sub Message → Cloud Function (Tracking Proxy) → Cloud Run Job (laxai-service-tracking)
```

## Infrastructure Setup

### Prerequisites

- Google Cloud Project with billing enabled
- Appropriate IAM permissions for Cloud Functions, Cloud Run, and Pub/Sub
- Service account with necessary roles

### Setup Steps

1. **Create Pub/Sub Topic**
   ```bash
   ./scripts/setup_tracking_pubsub.sh
   ```

2. **Deploy Cloud Function**
   ```bash
   gcloud builds submit . --config=deploy/cloudbuild/cloudbuild-trackingproxy-functions.yaml
   ```

## Usage

### Publishing Tracking Jobs

Send messages to the `tracking-jobs` Pub/Sub topic with the following format:

```json
{
  "action": "create",
  "tenant_id": "your-tenant-id",
  "custom_name": "optional-job-name",
  "task_id": "optional-unique-task-id",
  "tracking_params": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

### Canceling Jobs

```json
{
  "action": "cancel",
  "task_id": "task-id-to-cancel"
}
```

### Getting Job Status

The proxy stores job status in Firestore under the `tracking_runs` collection. You can query job status using the task_id.

## Configuration

### Environment Variables

- `GOOGLE_CLOUD_PROJECT`: GCP project ID (automatically set)
- `CLOUD_REGION`: GCP region (defaults to us-central1)

### Cloud Function Settings

- **Memory**: 512MB
- **Timeout**: 540 seconds (9 minutes)
- **Max Instances**: 10
- **Runtime**: Python 3.12

## Development

### Local Testing

1. Set up virtual environment:
   ```bash
   cd services/service_cr_tracking_proxy
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run tests:
   ```bash
   python -m pytest tests/ -v
   ```

### Dependencies

- `google-cloud-pubsub`: Pub/Sub client
- `google-cloud-run`: Cloud Run management
- `google-cloud-firestore`: Status storage
- `shared_libs`: Common utilities

## Monitoring

- Cloud Function logs are available in Cloud Logging
- Job execution status is tracked in Firestore
- Pub/Sub message delivery can be monitored in the GCP Console

## Troubleshooting

### Common Issues

1. **Function deployment fails**: Check IAM permissions for the service account
2. **Pub/Sub messages not processed**: Verify topic name and message format
3. **Cloud Run job fails**: Check Cloud Run job configuration and permissions

### Logs

View function logs:
```bash
gcloud functions logs read laxai-tracking-proxy --region=us-central1
```