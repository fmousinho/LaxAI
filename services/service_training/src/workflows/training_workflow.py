"""
Training workflow orchestration for LaxAI.

This module contains the core training workflow logic that can be used
by CLI, API, or other interfaces.
"""
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from common.google_storage import GCSPaths, get_storage
from config.logging_config import print_banner
from parameter_registry import parameter_registry
from scipy import datasets
from train_pipeline import TrainPipeline
from utils.cpu_memory import clear_cpu_memory, log_comprehensive_memory_stats
# Absolute imports
from utils.env_secrets import setup_environment_secrets

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

            # Limit datasets if specified
            if self.n_datasets_to_use and len(datasets) > self.n_datasets_to_use:
                datasets = datasets[:self.n_datasets_to_use]
                logger.info(f"Limited to {self.n_datasets_to_use} datasets: {datasets}")

            return datasets

        except Exception as e:
            logger.error(f"Error discovering datasets: {e}")
            raise

    def execute(self) -> Dict[str, Any]:
        """
        Execute the complete training workflow.

        Returns:
            Dictionary containing training results and metadata.
        """
        logger.info(f"--- Starting Training Workflow for Tenant: {self.tenant_id} ---")
        logger.info(f"Training configuration: {self.training_kwargs}")
        logger.info(f"Model configuration: {self.model_kwargs}")

        try:
            # Discover available datasets
            datasets = self.discover_datasets()

            if not datasets:
                logger.warning("No datasets found for training")
                return {
                    "status": "completed",
                    "datasets_found": 0,
                    "training_runs": [],
                    "message": "No datasets available for training"
                }

            # Execute training for each dataset
            training_results = []
            successful_runs = 0

            for dataset_name in datasets:
                try:
                    logger.info(f"Starting training for dataset: {dataset_name}")

                    result = self._run_training_for_dataset(dataset_name)
                    training_results.append(result)

                    if result.get("status") == "success":
                        successful_runs += 1

                except InterruptedError:
                    # Handle cancellation gracefully
                    logger.info(f"Training cancelled during dataset {dataset_name}")
                    training_results.append({
                        "dataset": dataset_name,
                        "status": "cancelled",
                        "error": "Training cancelled by user request"
                    })
                    break  # Stop processing remaining datasets

                except Exception as e:
                    logger.error(f"Training failed for dataset {dataset_name}: {e}")
                    training_results.append({
                        "dataset": dataset_name,
                        "status": "failed",
                        "error": str(e)
                    })

            # Aggregate steps completed across all dataset runs
            steps_completed_total = 0
            run_guids = []
            for r in training_results:
                try:
                    steps_completed_total += int(r.get("result", {}).get("steps_completed", 0))
                except Exception:
                    # Non-standard result entry
                    continue
                
                # Collect run_guids for cancellation
                if r.get("run_guid"):
                    run_guids.append(r["run_guid"])

            # Return summary
            return {
                "status": "completed",
                "datasets_found": len(datasets),
                "successful_runs": successful_runs,
                "total_runs": len(training_results),
                "steps_completed": steps_completed_total,
                "training_results": training_results,
                "run_guids": run_guids,  # Return all run_guids for cancellation
                "custom_name": self.custom_name
            }

        except InterruptedError:
            # Handle workflow-level cancellation
            logger.info("Training workflow cancelled at top level")
            return {
                "status": "cancelled",
                "datasets_found": 0,
                "successful_runs": 0,
                "total_runs": 0,
                "steps_completed": 0,
                "training_results": [],
                "custom_name": self.custom_name,
                "cancelled": True,
                "error": "Training workflow cancelled by user request"
            }
        except Exception as e:
            logger.error(f"Training workflow failed: {e}")
            raise

    def _run_training_for_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Run training pipeline for a specific dataset.

        Args:
            dataset_name: Name of the dataset to train on.

        Returns:
            Dictionary containing training result for this dataset.
        """
        try:
            # Combine all kwargs for TrainPipeline
            all_kwargs = {
                **self.training_kwargs,
                **self.model_kwargs,
                **self.eval_kwargs
            }

            # Remove duplicate pipeline_name if present
            if 'pipeline_name' in all_kwargs:
                logger.debug("Removing duplicate 'pipeline_name' from merged kwargs")
                all_kwargs.pop('pipeline_name')

            # Create training pipeline
            train_pipeline = TrainPipeline(
                tenant_id=self.tenant_id,
                verbose=self.verbose,
                save_intermediate=self.save_intermediate,
                pipeline_name=f"{self.pipeline_name}_{dataset_name}",
                **all_kwargs
            )

            # Extract run_guid immediately after pipeline creation (before run())
            pipeline_run_guid = train_pipeline.run_guid

            # Execute the training (pipeline handles cancellation automatically)
            result = train_pipeline.run(
                dataset_name=[dataset_name],
                resume_from_checkpoint=self.resume_from_checkpoint
            )

            return {
                "dataset": dataset_name,
                "status": "success",
                "result": result,
                "run_guid": pipeline_run_guid  # Return the GUID for cancellation
            }

        except InterruptedError:
            # Handle cancellation during dataset training
            logger.info(f"Dataset training cancelled for {dataset_name}")
            return {
                "dataset": dataset_name,
                "status": "cancelled",
                "error": "Dataset training cancelled by user request"
            }
        except Exception as e:
            logger.error(f"Training failed for dataset {dataset_name}: {e}")
            return {
                "dataset": dataset_name,
                "status": "failed",
                "error": str(e)
            }



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
