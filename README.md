LaxAI Training Service
======================

This repository provides an orchestrated training service for player re-identification and team classification.
It supports both a CLI for batch workflows and a FastAPI web API for starting and managing training jobs.

Highlights
 - End-to-end data preparation and training pipelines (dataprep → training → checkpointing)
 - WandB integration for experiment tracking and model artifact management
 - Hugging Face model support (e.g., DINOv3 backbone) with gated model token support
 - Affine-aware tracking utilities (AffineAwareByteTrack) to compensate camera motion

Quickstart (local)
-------------------

1. Create and activate a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements-base.txt
```

2. Provide secrets (create `.env` from `.env.example`):

```bash
cp .env.example .env
# Fill in WANDB_API_KEY and HUGGINGFACE_HUB_TOKEN in .env
```

3. Run the CLI (example):

```bash
PYTHONPATH=./src python src/scripts/train_all.py --tenant_id tenant1 --custom_name myrun --n_datasets_to_use 1
```

4. Run the API (development):

```bash
# From repo root
PYTHONPATH=./src python src/main.py
# Then use the API at http://localhost:8000
```

CLI reference
-------------

Primary script: `src/scripts/train_all.py`

Key flags (run `--help` for the full list):

```
usage: train_all.py [-h] [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE]
										[--learning-rate LEARNING_RATE] [--margin MARGIN]
										[--weight-decay WEIGHT_DECAY] [...]
										[--tenant_id TENANT_ID] [--frames FRAMES]
										[--verbose] [--save_intermediate]
										[--custom_name CUSTOM_NAME]
										[--resume_from_checkpoint]
										[--wandb_tags [WANDB_TAGS ...]]
										[--n_datasets_to_use N_DATASETS_TO_USE]

Run the full LaxAI Data Prep and Training Workflow.

options:
	-h, --help            show this help message and exit
	--num-epochs NUM_EPOCHS
												Number of training epochs (default: 50)
	--batch-size BATCH_SIZE
												Batch size for training (default: 64)
	--learning-rate LEARNING_RATE
												Learning rate for optimizer (default: 0.001)
	--margin MARGIN       Margin for triplet loss (default: 0.4)
	--tenant_id TENANT_ID The tenant ID for GCS.
	--frames FRAMES       Number of frames to extract per video.
	--verbose             Enable verbose pipeline logging.
	--save_intermediate   Save intermediate pipeline step results to GCS.
	--custom_name CUSTOM_NAME
												Custom name for the training run (used in wandb and logging).
	--resume_from_checkpoint
												Resume training from checkpoint if available.
	--wandb_tags [WANDB_TAGS ...]
												List of tags for wandb tracking (space-separated).
	--n_datasets_to_use N_DATASETS_TO_USE
												Limit number of discovered datasets to use for training (top-level param).
```

API reference
-------------

The FastAPI server exposes a small set of endpoints for starting and managing training jobs.

Base URL: http://localhost:8000

Endpoints:

- POST /api/v1/train — Start a training job. Accepts a JSON body matching the `TrainingRequest` schema.
- GET /api/v1/train/{task_id}/status — Query status for a started job.
- GET /api/v1/train/jobs — List all known jobs and statuses.
- DELETE /api/v1/train/{task_id} — Cancel a running job.
- GET /api/v1/train/pipelines — List currently active pipelines.

Example API payload (generated from parameter registry):

```json
{
	"tenant_id": "tenant1",
	"verbose": false,
	"custom_name": "api_run_20250825_152109_725eb37b",
	"resume_from_checkpoint": true,
	"wandb_tags": [
		"api"
	],
	"training_params": {
		"num_epochs": 50,
		"batch_size": 64,
		"learning_rate": 0.001,
		"margin": 0.4,
		"weight_decay": 0.0001,
		"lr_scheduler_patience": 3,
		"num_workers": 0,
		"prefetch_factor": 2,
		"embedding_dim": 512,
		"dropout_rate": 0.2,
		"input_height": 120,
		"input_width": 80
	},
	"model_params": {}
}
```

See `src/api/example_client.py` for a working Python client example.

Configuration & Secrets
-----------------------

Required for normal operation:

- `WANDB_API_KEY` — WandB API key for experiment tracking and artifact uploads.
- `HUGGINGFACE_HUB_TOKEN` — Required to download gated Hugging Face models like DINOv3.

Optional but recommended in some environments:

- `GOOGLE_CLOUD_PROJECT` — Project id used to read secrets from Google Secret Manager.
- `GOOGLE_APPLICATION_CREDENTIALS` — Path to a service account JSON key (used for Secret Manager and GCS operations).

For local development, create a `.env` file (copy from `.env.example`) and fill the values. The runtime will load secrets from the environment, then from `.env`, and finally (if `GOOGLE_CLOUD_PROJECT` is set) from Secret Manager.

Troubleshooting
---------------

- If the FastAPI app fails to import with a Secret error, ensure `WANDB_API_KEY` and `HUGGINGFACE_HUB_TOKEN` are present in your environment or `.env`.
- WandB requires a 40-character API key. If using local dummy keys for testing, set `WANDB_MODE=offline` to avoid network verification.
- If you want Secret Manager fallback, ensure `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` are set and the service account has `secretmanager.secretAccessor`.

Development & Testing
---------------------

Run unit tests with pytest from the repo root:

```bash
PYTHONPATH=./src ./.venv31211/bin/python -m pytest -q
```

Create `.env.example` (already provided) and do not commit real secrets. Use CI secrets or Secret Manager for the real keys.

Contributing
------------

See `CONTRIBUTING.md` in the `documentation/` folder for guidelines.
