"""
Cloud Run Job deployment configuration and helper scripts.

This module provides utilities for deploying and managing
Cloud Run Jobs for GPU training workloads.
"""

# Cloud Run Job configuration template
CLOUD_RUN_JOB_CONFIG = {
    "apiVersion": "run.googleapis.com/v1",
    "kind": "Job",
    "metadata": {
        "name": "laxai-worker",
        "labels": {
            "cloud.googleapis.com/location": "us-central1"
        }
    },
    "spec": {
        "template": {
            "spec": {
                "template": {
                    "spec": {
                        "restartPolicy": "Never",
                        "taskTimeoutSeconds": 25200,  # 7 hours
                        "containers": [{
                            "name": "training-worker",
                            "image": "gcr.io/{PROJECT_ID}/laxai-worker:latest",
                            
                            "env": [
                                {"name": "GOOGLE_CLOUD_PROJECT", "value": "{PROJECT_ID}"},
                                {"name": "TRAINING_JOBS_SUBSCRIPTION", "value": "training-jobs-sub"},
                                {"name": "FIRESTORE_ENABLED", "value": "true"},
                                {"name": "WORKER_TIMEOUT", "value": "25200"},
                                {"name": "MAX_CONCURRENT_JOBS", "value": "1"}
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": "4",
                                    "memory": "16Gi",
                                    "nvidia.com/gpu": "1"
                                },
                                "requests": {
                                    "cpu": "2",
                                    "memory": "8Gi",
                                    "nvidia.com/gpu": "1"
                                }
                            }
                        }],
                        "nodeSelector": {
                            "cloud.google.com/gke-accelerator": "nvidia-tesla-t4"
                        }
                    }
                }
            }
        }
    }
}

# Deployment commands
DEPLOYMENT_COMMANDS = {
        "build_image": [
            "docker build -f docker/worker/Dockerfile.worker -t us-central1-docker.pkg.dev/{PROJECT_ID}/laxai-repo/laxai-worker:latest ."
        ],
    "push_image": [
            "docker push us-central1-docker.pkg.dev/{PROJECT_ID}/laxai-repo/laxai-worker:latest"
    ],
    "create_subscription": [
        "gcloud pubsub subscriptions create training-jobs-sub --topic=training-jobs"
    ],
    "deploy_job": [
        "gcloud run jobs replace cloud-run-job.yaml --region=us-central1"
    ],
    "execute_job": [
        "gcloud run jobs execute laxai-worker --region=us-central1"
    ]
}

# Required IAM permissions for the Cloud Run Job service account
REQUIRED_PERMISSIONS = [
    "pubsub.subscriptions.consume",
    "pubsub.messages.ack",
    "datastore.entities.get",
    "datastore.entities.update",
    "logging.logEntries.create",
    "storage.objects.create",
    "storage.objects.get"
]
