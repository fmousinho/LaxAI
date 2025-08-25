# Cloud Function deployment configuration

# Cloud Functions runtime settings
functions_framework_python_version = "3.11"
runtime = "python311"

# Environment variables for Cloud Function
environment_variables = {
    "GOOGLE_CLOUD_PROJECT" = var.project_id
    "TRAINING_JOBS_TOPIC" = var.pubsub_topic_name
    "FIRESTORE_ENABLED" = "true"
    "LOG_LEVEL" = "INFO"
}

# Resource settings for Cloud Function
available_memory_mb = 512
timeout_seconds = 60
max_instances = 10

# IAM permissions required for the Cloud Function
required_permissions = [
    "pubsub.topics.publish",
    "datastore.entities.create",
    "datastore.entities.get", 
    "datastore.entities.update",
    "datastore.entities.delete",
    "logging.logEntries.create"
]
