"""
Training workflow orchestration for LaxAI.

This module contains the core training workflow logic that can be used
by CLI, API, or other interfaces.
"""
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from shared_libs.config.logging_config import print_banner
from parameter_registry import parameter_registry
from scipy import datasets
from train_pipeline import TrainPipeline

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.utils.cpu_memory import (clear_cpu_memory,
                                          log_comprehensive_memory_stats)
# Absolute imports
from shared_libs.utils.env_secrets import setup_environment_secrets

# Enable MPS fallback for unsupported operations, as recommended by PyTorch.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logger = logging.getLogger(__name__)


class TrainingWorkflow:
    """
    Orchestrates the end-to-end training process for LaxAI.

    This class handles the complete workflow from dataset discovery
    to model training execution.
    """

    def __init__(self,
                 tenant_id: str,
                 verbose: bool = True,
                 save_intermediate: bool = True,
                 custom_name: str = "training_workflow_run",
                 resume_from_checkpoint: bool = True,
                 wandb_tags: Optional[List[str]] = None,
                 training_kwargs: Optional[Dict[str, Any]] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 pipeline_name: Optional[str] = "default",
                 n_datasets_to_use: Optional[int] = None,
                 eval_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the training workflow.

        Args:
            tenant_id: The tenant ID for GCS operations.
            verbose: Enable verbose logging for pipelines.
            save_intermediate: Save intermediate pipeline results to GCS.
            custom_name: Custom name for the training run.
            resume_from_checkpoint: Resume training from checkpoint if available.
            wandb_tags: List of tags for wandb tracking.
            training_kwargs: Dictionary of training parameters.
            model_kwargs: Dictionary of model parameters.
            pipeline_name: Unique name for the pipeline.
            n_datasets_to_use: Limit number of datasets to use.
            eval_kwargs: Dictionary of evaluation parameters.
        """
        self.tenant_id = tenant_id
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.custom_name = custom_name
        self.resume_from_checkpoint = resume_from_checkpoint
        self.wandb_tags = wandb_tags or []
        self.training_kwargs = training_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.pipeline_name = pipeline_name
        self.n_datasets_to_use = n_datasets_to_use
        self.eval_kwargs = eval_kwargs or {}

    def discover_datasets(self) -> List[str]:
        """
        Find all available training datasets for the tenant.

        Returns:
            List of dataset paths available for training.
        """
        try:
            storage = get_storage(self.tenant_id)
            path_manager = GCSPaths()
            dataset_path = path_manager.get_path("datasets_root")

    
            datasets = []

            try:
                datasets = storage.list_blobs(
                    prefix=dataset_path,
                    delimiter='/',
                    exclude_prefix_in_return=True
                    )
                datasets = [d.rstrip('/') for d in datasets]

                logger.info(f"Found {len(datasets)} datasets: {datasets}")

            except Exception as e:
                logger.warning(f"Could not list datasets from {dataset_path}: {e}")
                datasets = []

            # Limit datasets if specified (0 means use all datasets)
            logger.info(f"n_datasets_to_use parameter: {self.n_datasets_to_use}")
            if self.n_datasets_to_use and self.n_datasets_to_use > 0 and self.n_datasets_to_use < len(datasets):
                original_count = len(datasets)
                datasets = datasets[:self.n_datasets_to_use]
                logger.info(f"🎯 LIMITED: Reduced from {original_count} to {self.n_datasets_to_use} datasets: {datasets}")
            elif self.n_datasets_to_use and self.n_datasets_to_use > 0:
                logger.info(f"⚠️  Requested {self.n_datasets_to_use} datasets, but only {len(datasets)} available")
            else:
                logger.info(f"📋 Using all {len(datasets)} datasets (n_datasets_to_use={self.n_datasets_to_use})")

            return datasets

        except Exception as e:
            logger.error(f"Error discovering datasets: {e}")
            raise

    def execute(self) -> Dict[str, Any]:
        """Execute training using a SINGLE TrainPipeline over all datasets.

        The previous implementation created one pipeline per dataset and aggregated
        their results. The TrainPipeline already supports multi-dataset mode by
        accepting a list of dataset names. We now:
        - Discover all datasets (optionally truncated by n_datasets_to_use)
        - Instantiate exactly one TrainPipeline whose pipeline_name (and external
          identity) is self.pipeline_name
        - Pass either a single dataset string or list of dataset strings to .run()
        - Return a flattened summary without per-dataset training_results
        """
        logger.info(f"--- Starting Training Workflow (single-pipeline) Tenant: {self.tenant_id} ---")
        logger.info(f"Training configuration: {self.training_kwargs}")
        logger.info(f"Model configuration: {self.model_kwargs}")

        try:
            datasets = self.discover_datasets()
            if not datasets:
                logger.warning("No datasets found for training")
                return {
                    "status": "completed",
                    "datasets_found": 0,
                    "steps_completed": 0,
                    "run_id": self.pipeline_name,
                    "pipeline_name": self.pipeline_name,
                    "run_guids": [],
                    "message": "No datasets available for training",
                    "custom_name": self.custom_name,
                    "dataset_mode": "none"
                }

            dataset_mode = "multi" if len(datasets) > 1 else "single"

            # Merge kwargs for TrainPipeline (avoid duplicate pipeline_name)
            all_kwargs = {**self.training_kwargs, **self.model_kwargs, **self.eval_kwargs}
            all_kwargs.pop("pipeline_name", None)

            # Create single pipeline; override run_guid with external identity (task_id/pipeline_name)
            pipeline_identity = self.pipeline_name or "training_run"
            train_pipeline = TrainPipeline(
                tenant_id=self.tenant_id,
                verbose=self.verbose,
                save_intermediate=self.save_intermediate,
                pipeline_name=pipeline_identity,  # Also used as identity
                run_guid=pipeline_identity,
                **all_kwargs
            )

            # Execute once over all datasets (list or single item)
            dataset_arg = datasets if dataset_mode == "multi" else datasets[0]
            pipeline_result = train_pipeline.run(
                dataset_name=dataset_arg,
                resume_from_checkpoint=self.resume_from_checkpoint,
                wandb_run_tags=self.wandb_tags,
                custom_name=self.custom_name,
            )

            status = pipeline_result.get("status", "unknown")
            steps_completed = int(pipeline_result.get("steps_completed", 0))

            return {
                "status": "completed" if status == "completed" else status,
                "datasets_found": len(datasets),
                "steps_completed": steps_completed,
                "run_id": pipeline_identity,
                "pipeline_name": pipeline_identity,
                "run_guids": [train_pipeline.run_guid],  # type: ignore[attr-defined]
                "custom_name": self.custom_name,
                "dataset_mode": dataset_mode,
                "pipeline_result": pipeline_result,
            }

        except InterruptedError:
            logger.info("Training workflow cancelled (single-pipeline mode)")
            return {
                "status": "cancelled",
                "datasets_found": 0,
                "steps_completed": 0,
                "run_id": self.pipeline_name,
                "pipeline_name": self.pipeline_name,
                "run_guids": [self.pipeline_name],
                "custom_name": self.custom_name,
                "dataset_mode": "unknown",
                "error": "Training workflow cancelled by user request"
            }
        except Exception as e:
            logger.error(f"Training workflow failed: {e}")
            return {
                "status": "failed",
                "datasets_found": 0,
                "steps_completed": 0,
                "run_id": self.pipeline_name,
                "pipeline_name": self.pipeline_name,
                "run_guids": [self.pipeline_name],
                "custom_name": self.custom_name,
                "dataset_mode": "unknown",
                "error": str(e)
            }

    # Removed _run_training_for_dataset: single pipeline now handles all datasets.



def train_workflow(tenant_id: str, **kwargs):
    """
    Convenience function for running training workflow.

    This function provides a simple interface for running the training workflow
    with keyword arguments, similar to the original train_all.train() function.

    Args:
        tenant_id: The tenant ID for GCS operations.
        **kwargs: Additional keyword arguments for TrainingWorkflow.

    Returns:
        Dictionary containing training results.
    """
    workflow = TrainingWorkflow(tenant_id=tenant_id, **kwargs)
    return workflow.execute()
