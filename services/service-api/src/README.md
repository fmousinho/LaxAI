# LaxAI Training API

A FastAPI-based REST API for managing LaxAI machine learning training pipelines.

## Features

- **Asynchronous Training**: Start training jobs in the background and track their progress
- **Flexible Configuration**: Support all training and model parameters via Pydantic schemas
- **Real-time Status**: Monitor training progress and view job statuses
- **RESTful Design**: Clean API endpoints for all operations
- **Auto-documentation**: Swagger UI and ReDoc documentation

## API Endpoints

### Training Operations

- `POST /api/v1/train` - Start a new training job
- `GET /api/v1/train/{task_id}/status` - Get training job status
- `GET /api/v1/train/jobs` - List all training jobs
- `DELETE /api/v1/train/{task_id}` - Cancel a training job

### System Operations

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

## Getting Started

### 1. Start the API Server

From the `src` directory:

```bash
python api/run_api.py
```

Or directly:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 2. View API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Example Usage

#### Start a Training Job

```bash
curl -X POST "http://localhost:8000/api/v1/train" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant1",
    "custom_name": "api_test_run",
    "verbose": true,
    "training_kwargs": {
      "num_epochs": 10,
      "batch_size": 32,
      "learning_rate": 0.001,
      "margin": 0.4
    },
    "model_kwargs": {
      "embedding_dim": 256,
      "use_cbam": true
    }
  }'
```

#### Check Training Status

```bash
curl "http://localhost:8000/api/v1/train/{task_id}/status"
```

#### List All Jobs

```bash
curl "http://localhost:8000/api/v1/train/jobs"
```

## Request Schema

### Training Request

```json
{
  "tenant_id": "string", // Required: GCS tenant ID
  "frames": 20, // Optional: frames per video
  "verbose": true, // Optional: enable verbose logging
  "save_intermediate": true, // Optional: save intermediate results
  "custom_name": "train_all_run", // Optional: custom run name
  "resume_from_checkpoint": true, // Optional: resume from checkpoint
  "wandb_tags": ["tag1", "tag2"], // Optional: WandB tags
  "training_kwargs": {
    // Optional: training parameters
    "num_epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
    "min_images_per_player": 2,
    "margin": 0.4,
    "weight_decay": 0.0001,
    "margin_decay_rate": 0.99,
    "margin_change_threshold": 0.01,
    "lr_scheduler_patience": 3,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_min_lr": 0.0000001,
    "num_workers": 8,
    "prefetch_factor": 2,
    "force_pretraining": false
  },
  "model_kwargs": {
    // Optional: model parameters
    "embedding_dim": 512,
    "dropout_rate": 0.2,
    "use_cbam": true,
    "attention_layers": ["layer1", "layer2"]
  }
}
```

### Response Schema

```json
{
  "status": "accepted", // Job status
  "task_id": "uuid-string", // Unique task identifier
  "message": "Training job started...",
  "progress": {
    "status": "pending",
    "current_epoch": null,
    "total_epochs": null,
    "current_loss": null,
    "best_loss": null,
    "message": "Training job queued",
    "datasets_found": null,
    "datasets_processed": null,
    "logs": []
  }
}
```

## Training Parameters

### Training Configuration (`training_kwargs`)

| Parameter                 | Type  | Description                        |
| ------------------------- | ----- | ---------------------------------- |
| `num_epochs`              | int   | Number of training epochs          |
| `batch_size`              | int   | Training batch size                |
| `learning_rate`           | float | Learning rate for training         |
| `early_stopping_patience` | int   | Early stopping patience            |
| `min_images_per_player`   | int   | Minimum images per player          |
| `margin`                  | float | Triplet loss margin                |
| `weight_decay`            | float | L2 regularization weight decay     |
| `margin_decay_rate`       | float | Decay rate for triplet loss margin |
| `margin_change_threshold` | float | Threshold for margin changes       |
| `lr_scheduler_patience`   | int   | Learning rate scheduler patience   |
| `lr_scheduler_factor`     | float | Learning rate reduction factor     |
| `lr_scheduler_min_lr`     | float | Minimum learning rate              |
| `num_workers`             | int   | Number of DataLoader workers       |
| `prefetch_factor`         | int   | DataLoader prefetch factor         |
| `force_pretraining`       | bool  | Force use of pretrained weights    |

### Model Configuration (`model_kwargs`)

| Parameter          | Type      | Description                      |
| ------------------ | --------- | -------------------------------- |
| `embedding_dim`    | int       | Dimension of output embeddings   |
| `dropout_rate`     | float     | Dropout rate in embedding layer  |
| `use_cbam`         | bool      | Use CBAM attention modules       |
| `attention_layers` | list[str] | ResNet layers for CBAM attention |

## Job Status Values

- `pending` - Job queued but not started
- `running` - Job currently executing
- `completed` - Job finished successfully
- `failed` - Job encountered an error
- `cancelled` - Job was cancelled

## Python Client Example

See `src/api/example_client.py` for a complete Python client example:

```python
from api.example_client import LaxAIClient

client = LaxAIClient("http://localhost:8000")
response = await client.start_training({
    "tenant_id": "tenant1",
    "training_kwargs": {"num_epochs": 10}
})
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `404` - Task not found
- `422` - Validation error
- `500` - Internal server error

Error responses include detailed information:

```json
{
  "error": "error_type",
  "message": "Human readable error message",
  "details": {
    "additional": "error context"
  }
}
```

## Development

### Dependencies

All required dependencies are listed in `requirements.txt`:

- `fastapi[all]` - FastAPI framework with all extras
- `uvicorn` - ASGI server
- Other LaxAI dependencies

### Running in Development

```bash
# From the src directory
python main.py
```

The server will reload automatically when code changes are detected.

## Production Deployment

For production deployment, consider:

1. **Process Management**: Use a proper process manager (systemd, supervisor, etc.)
2. **Reverse Proxy**: Deploy behind nginx or similar
3. **Database**: Replace in-memory job storage with Redis or database
4. **Authentication**: Add API key or OAuth authentication
5. **Monitoring**: Add application monitoring and logging
6. **Scaling**: Consider containerization with Docker/Kubernetes

Example production command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```
