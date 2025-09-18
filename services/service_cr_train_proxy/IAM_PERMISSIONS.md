# IAM Permissions for Cloud Function Training Proxy

The Cloud Function uses the `googlebatch` service account and requires the following permissions to execute training jobs and handle cancellation:

## Required Permissions

### 1. Cloud Run Jobs (Creation and Execution)
- `run.jobs.run` - Execute existing Cloud Run Jobs with argument overrides
- `run.jobs.get` - Read job configuration details
- `run.executions.get` - Read execution status (optional, for enhanced status)
- `run.executions.list` - List executions for a job (optional, for status)

### 2. Long-Running Operations (Cancellation)
- `run.operations.get` - Read operation status
- `run.operations.cancel` - Cancel running operations (LRO cancellation)

### 3. Firestore (Mapping Storage)
- `datastore.entities.create` - Create mapping documents
- `datastore.entities.get` - Read mapping documents
- `datastore.entities.update` - Update mapping status
- `datastore.entities.delete` - Clean up old mappings (optional)

### 4. Pub/Sub (Message Processing)
- `pubsub.messages.ack` - Acknowledge processed messages
- `pubsub.subscriptions.consume` - Consume messages from subscription

## Recommended IAM Roles

Instead of individual permissions, use these predefined roles:

1. **Cloud Run Developer**: `roles/run.developer`
   - Includes job execution and operation management permissions
   
2. **Cloud Datastore User**: `roles/datastore.user`
   - Includes Firestore read/write permissions
   
3. **Pub/Sub Subscriber**: `roles/pubsub.subscriber`
   - Includes message consumption permissions

## Grant Commands

```bash
# Set your project ID
PROJECT_ID="laxai-466119"

# Grant Cloud Run Developer role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:googlebatch@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.developer"

# Grant Cloud Datastore User role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:googlebatch@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/datastore.user"

# Grant Pub/Sub Subscriber role (if not already granted)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:googlebatch@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.subscriber"
```

## Verify Permissions

After granting permissions, verify with:

```bash
# Check current IAM policy for the service account
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:googlebatch@$PROJECT_ID.iam.gserviceaccount.com"
```

## Security Notes

- These permissions are scoped to the project level
- The `googlebatch` service account should only be used by Cloud Functions
- Consider using Workload Identity Federation for production environments
- Regularly audit and rotate service account keys if using key-based authentication