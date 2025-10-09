# IAM Permissions for Create Dataset from GML Cloud Function

The Cloud Function requires permissions to read/write GCS files and process graph data.

## Required Permissions

### 1. Cloud Storage (GCS Access)
- `storage.objects.get` - Read GML files and track crops
- `storage.objects.list` - List files in folders
- `storage.objects.create` - Create dataset folders and copy files
- `storage.objects.delete` - Optional, for cleanup

## Recommended IAM Roles

1. **Storage Object Admin**: `roles/storage.objectAdmin`
   - Full control over GCS objects

## Grant Commands

```bash
# Set your project ID
PROJECT_ID="laxai-466119"
SERVICE_ACCOUNT="laxai-466119@appspot.gserviceaccount.com"

# Grant Storage Object Admin role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/storage.objectAdmin"
```