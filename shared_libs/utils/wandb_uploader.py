#!/usr/bin/env python3
"""
Utility script to upload the detection model to Weights & Biases as an artifact.

This script performs the following actions:
1. Reads W&B and detection model configuration.
2. Downloads the specified detection model from Google Cloud Storage.
3. Initializes a new W&B run.
4. Creates a versioned W&B artifact for the model.
5. Uploads the model file to the artifact.
6. Finishes the run and cleans up temporary files.
"""
import logging
import os
import sys
# Suppress WandB Scope.user deprecation warning
import warnings

from config.all_config import wandb_config

import wandb

warnings.filterwarnings(
    "ignore", message=r".*The `Scope\.user` setter is deprecated.*", category=DeprecationWarning
)


# --- Path Setup ---
# Add the project root to the Python path to allow for absolute imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


logger = logging.getLogger(__name__)


REGISTRY_NAME = "Model"
COLLECTION_NAME = "Detections"
MODEL_PATH = "data/common-models-detection_latest.pth"


def upload_detection_model_to_wandb(
    model_name: str = "detection-model-rfdetr",
    model_type: str = "model",
    description: str = "RF-DETR detection model for player detection.",
):
    """
    Downloads the detection model from GCS and uploads it as an artifact to W&B.
    """
    if not wandb_config.enabled:
        logger.info("W&B logging is disabled in the configuration. Skipping upload.")
        return

    wandb_api_key = os.getenv("WANDB_API_KEY")

    if not wandb_api_key:
        logger.error(
            "WANDB_API_KEY not found in environment variables or .env file. Skipping upload."
        )
        return
    else:
        logger.info("WANDB_API_KEY found in environment variables or .env file.")

    try:
        logger.info("Logging into wandb")

        wandb.login(key=wandb_api_key)

        logger.info(
            f"Initializing project {wandb_config.project} with entity {wandb_config.entity}"
        )
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.team,
            job_type="upload-artifact",
            tags=["detection-model", "artifact", "rfdetr"],
            name=f"upload_{model_name}",
        )
        if not run:
            raise RuntimeError("Failed to initialize W&B run.")

        logger.info(f"Creating W&B artifact with name: '{model_name}'")

        artifact = run.log_artifact(artifact_or_path=MODEL_PATH, name=model_name, type=model_type)

        run.link_artifact(
            artifact=artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
        )

        logger.info("Logging artifact to W&B...")
        run.log_artifact(artifact)
        run.finish()
        logger.info("✅ Successfully uploaded detection model to W&B.")

    except Exception as e:
        logger.error(f"An error occurred during W&B upload: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    upload_detection_model_to_wandb()
