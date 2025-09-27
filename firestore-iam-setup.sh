# Required IAM roles for service_api Cloud Run service:
# - Cloud Datastore User (roles/datastore.user) - for Firestore access
# - Pub/Sub Publisher (roles/pubsub.publisher) - for job queuing
# - Storage Object Admin (roles/storage.objectAdmin) - for GCS access

# Grant Firestore permissions to your Cloud Run service account:
gcloud projects add-iam-policy-binding laxai-466119 \
  --member='serviceAccount:service_api@laxai-466119.iam.gserviceaccount.com' \
  --role='roles/datastore.user'

# For service_tracking (GPU service):
gcloud projects add-iam-policy-binding laxai-466119 \
  --member='serviceAccount:service_tracking@laxai-466119.iam.gserviceaccount.com' \
  --role='roles/datastore.user'
