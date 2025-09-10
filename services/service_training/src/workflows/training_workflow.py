"""
Training workflow for service_training module.

This module provides the TrainingWorkflow class that integrates with
Google Storage for datasets and WandB for experiment tracking.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

from shared_libs.utils.env_secrets import setup_environment_secrets

# Setup environment secrets
setup_environment_secrets()

logger = logging.getLogger(__name__)


class TrainingWorkflow:
    """
    End-to-end training workflow class.
    """

    def __init__(
        self,
        tenant_id: str,
        verbose: bool = True,
        save_intermediate: bool = True,
        custom_name: str = "training_workflow_run",
        resume_from_checkpoint: bool = True,
        wandb_tags: Optional[List[str]] = None,
        training_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        pipeline_name: str = "default",
        n_datasets_to_use: Optional[int] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the training workflow.

        Args:
            tenant_id: The tenant ID for GCS operations.
            verbose: Enable verbose logging.
            save_intermediate: Save intermediate results.
            custom_name: Custom name for the training run.
            resume_from_checkpoint: Resume from checkpoint if available.
            wandb_tags: List of tags for wandb tracking.
            training_kwargs: Dictionary of training parameters.
            model_kwargs: Dictionary of model parameters.
            pipeline_name: Unique name for the pipeline.
            n_datasets_to_use: Number of datasets to use.
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

    def execute(self) -> Dict[str, Any]:
        """
        Execute the training workflow.

        Returns:
            Dictionary containing training results.
        """
        try:
            # Import training components
            from train_pipeline import TrainPipeline

            # Create and run the training pipeline
            pipeline = TrainPipeline(
                tenant_id=self.tenant_id,
                verbose=self.verbose,
                save_intermediate=self.save_intermediate,
                pipeline_name=self.pipeline_name,
                **self.training_kwargs
            )

            # Execute the pipeline
            results = pipeline.execute()

            return {
                "status": "success",
                "results": results,
                "custom_name": self.custom_name,
                "tenant_id": self.tenant_id
            }

        except Exception as e:
            logger.error(f"Training workflow failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "custom_name": self.custom_name,
                "tenant_id": self.tenant_id
            }
