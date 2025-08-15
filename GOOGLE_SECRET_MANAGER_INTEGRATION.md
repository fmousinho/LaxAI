# Google Secret Manager Integration

The `env_or_colab.py` module now includes comprehensive Google Secret Manager integration for secure credential management across all environments.

## Overview

The integration provides automatic fallback to Google Secret Manager when environment variables or other credential sources are not available. This ensures secure credential management in production environments while maintaining flexibility for development.

## Features

- ✅ **Automatic Project Detection**: Detects GCP project from environment variables or metadata service
- ✅ **Multi-Environment Support**: Works in local development, Google Colab, and all GCP services
- ✅ **Graceful Fallbacks**: Falls back to Secret Manager when primary credential sources fail
- ✅ **Security Best Practices**: Doesn't log sensitive credential values
- ✅ **Configurable Secret Names**: Supports multiple naming conventions

## Setup

### 1. Install Dependencies

```bash
pip install google-cloud-secret-manager
```

### 2. Enable Secret Manager API

```bash
gcloud services enable secretmanager.googleapis.com
```

### 3. Create Secrets

Create secrets for your application credentials:

```bash
# Create WandB API key secret
echo "your-wandb-api-key-here" | gcloud secrets create wandb-api-key --data-file=-

# Create OpenAI API key secret (if needed)
echo "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-

# Create HuggingFace token secret (if needed)
echo "your-hf-token" | gcloud secrets create huggingface-token --data-file=-
```

### 4. Grant Access

Grant your service account access to the secrets:

```bash
# For a specific service account
gcloud secrets add-iam-policy-binding wandb-api-key \
  --member='serviceAccount:your-service-account@project.iam.gserviceaccount.com' \
  --role='roles/secretmanager.secretAccessor'

# For Compute Engine default service account
gcloud secrets add-iam-policy-binding wandb-api-key \
  --member='serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com' \
  --role='roles/secretmanager.secretAccessor'
```

## Usage

### Automatic Integration

The Secret Manager integration works automatically when you import the module:

```python
# This automatically sets up credentials from all available sources
from utils.env_or_colab import load_env_or_colab

# Environment is now configured with credentials from:
# 1. Environment variables (if available)
# 2. .env files (local development)
# 3. Google Colab userdata (if in Colab)
# 4. Google Secret Manager (as fallback)
```

### Manual Secret Retrieval

You can also manually retrieve secrets:

```python
from utils.env_or_colab import get_secret_from_manager

# Retrieve a secret
api_key = get_secret_from_manager('wandb-api-key')
if api_key:
    print(f"Retrieved API key: {api_key[:8]}...")
else:
    print("Secret not found")

# Retrieve with specific project
api_key = get_secret_from_manager('wandb-api-key', project_id='my-project')
```

### Environment-Specific Setup

You can also set up secrets for specific environment variables:

```python
from utils.env_or_colab import setup_secret_from_manager

# Set WANDB_API_KEY from Secret Manager
success = setup_secret_from_manager('WANDB_API_KEY', 'wandb-api-key')
if success:
    print("WandB API key loaded from Secret Manager")
```

## Environment Behavior

### Local Development
1. Loads `.env` file first
2. If credentials missing, tries Secret Manager (if GCP credentials available)
3. Logs warnings for missing credentials

### Google Colab
1. Loads from Colab userdata first
2. Falls back to Secret Manager for missing credentials
3. Requires GCP project configuration for Secret Manager

### Google Cloud Platform
1. Uses default GCP credentials automatically
2. Detects project ID from metadata service
3. Retrieves all configured secrets from Secret Manager
4. Supports all GCP services (Cloud Run, App Engine, GKE, etc.)

## Supported Secrets

The integration automatically attempts to retrieve these secrets:

| Environment Variable | Secret Name (Primary) | Secret Name (Alternative) |
|---------------------|----------------------|-------------------------|
| `WANDB_API_KEY` | `wandb-api-key` | `WANDB_API_KEY` |
| `OPENAI_API_KEY` | `openai-api-key` | `OPENAI_API_KEY` |
| `HUGGINGFACE_TOKEN` | `huggingface-token` | `HUGGINGFACE_TOKEN` |

You can add more secrets by modifying the `secrets_to_check` list in `setup_gcp_credentials()`.

## Project ID Detection

The integration automatically detects your GCP project ID from:

1. **Environment Variables**: `GOOGLE_CLOUD_PROJECT`, `GCLOUD_PROJECT`, `GCP_PROJECT`
2. **Metadata Service**: Available in all GCP compute environments
3. **Manual Configuration**: Set `GOOGLE_CLOUD_PROJECT` explicitly

## Security Considerations

- ✅ **No Credential Logging**: Secret values are never logged
- ✅ **Least Privilege**: Only requests `secretAccessor` role
- ✅ **Environment Isolation**: Different secrets for different environments
- ✅ **Graceful Degradation**: Application continues if secrets unavailable

## Troubleshooting

### Common Issues

1. **"Project ID not found"**
   ```bash
   # Set project ID explicitly
   export GOOGLE_CLOUD_PROJECT=your-project-id
   ```

2. **"Permission denied"**
   ```bash
   # Check IAM permissions
   gcloud secrets add-iam-policy-binding secret-name \
     --member='serviceAccount:your-sa@project.iam.gserviceaccount.com' \
     --role='roles/secretmanager.secretAccessor'
   ```

3. **"Secret not found"**
   ```bash
   # List existing secrets
   gcloud secrets list
   
   # Create missing secret
   echo "your-secret-value" | gcloud secrets create secret-name --data-file=-
   ```

### Debug Mode

Enable debug logging to see detailed credential loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from utils.env_or_colab import load_env_or_colab
# Will show detailed logs of credential loading process
```

## Testing

Run the test script to verify your Secret Manager setup:

```bash
python test_secret_manager.py
```

This will test:
- Secret Manager availability
- GCP project detection
- Secret retrieval
- Full environment setup

## Best Practices

1. **Use Descriptive Secret Names**: `wandb-api-key` instead of `secret1`
2. **Set Appropriate IAM**: Grant minimal required permissions
3. **Use Different Secrets for Different Environments**: `wandb-api-key-dev`, `wandb-api-key-prod`
4. **Enable Secret Versioning**: Keep track of secret changes
5. **Regular Secret Rotation**: Update secrets periodically

## Integration with Training Pipeline

The training pipeline automatically benefits from Secret Manager integration:

```python
# In your training script
from utils.env_or_colab import load_env_or_colab

# This will automatically load WandB API key from Secret Manager if needed
# No changes required to existing training code!

import wandb
wandb.init(project="lacrosse-player-tracking")  # Just works!
```

## Cloud Deployment

When deploying to GCP services:

### Cloud Run
```yaml
# No additional configuration needed
# Secret Manager integration works automatically
```

### App Engine
```yaml
# app.yaml
env_variables:
  GOOGLE_CLOUD_PROJECT: your-project-id
```

### GKE with Workload Identity
```yaml
# Workload Identity automatically provides credentials
# Secret Manager integration works out of the box
```

This integration makes your LaxAI training pipeline production-ready with enterprise-grade credential management!
