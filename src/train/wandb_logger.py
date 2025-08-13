import os
import logging
import tempfile
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional, List, Callable
import torch
import numpy as np
from PIL import Image
from utils.env_or_colab import load_env_or_colab

from config.all_config import wandb_config

logger = logging.getLogger(__name__)

# Try to import wandb, handle gracefully if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Training will continue without wandb logging.")

CHECKPOINT_NAME = "checkpoint"


def requires_wandb_enabled(func: Callable) -> Callable:
    """Decorator to check if wandb is enabled before executing method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            logger.debug(f"Wandb not enabled, skipping {func.__name__}")
            return None
        return func(self, *args, **kwargs)
    return wrapper


def requires_wandb_initialized(func: Callable) -> Callable:
    """Decorator to check if wandb is enabled and initialized before executing method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.enabled or not self.initialized or self.run is None:
            logger.warning(f"Wandb logging is not enabled or initialized for {func.__name__}")
            return None
        return func(self, *args, **kwargs)
    return wrapper


def safe_wandb_operation(default_return=None):
    """Decorator to safely execute wandb operations with error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to execute {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator


class WandbLogger:

    """
    Utility class for Weights & Biases logging integration.
    Handles experiment tracking, metrics logging, and model artifacts.
    """

    def __init__(self, enabled: bool = wandb_config.enabled):
        """
        Initialize the wandb logger.
        
        Args:
            enabled: Override config setting for wandb logging
        """
        self.enabled = enabled
        self.run = None
        self.initialized = False
        
        if not WANDB_AVAILABLE:
            self.enabled = False
            logger.warning("wandb package not available, disabling wandb logging")
    
    def _get_api_key(self) -> Optional[str]:
        """Get and validate wandb API key."""
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            logger.error("WANDB_API_KEY environment variable not found")
            return None
        return api_key
    
    def _login_and_get_api(self) -> Optional[object]:
        """Login to wandb and return API object."""
        api_key = self._get_api_key()
        if not api_key:
            return None
        
        wandb.login(key=api_key)
        return wandb.Api()
    
    def _construct_artifact_path(self, artifact_name: str, version: str = "latest") -> str:
        """Construct standardized artifact path."""
        return f"{wandb_config.team}/{wandb_config.project}/{artifact_name}:{version}"
    
    @requires_wandb_enabled
    @safe_wandb_operation(default_return=False)
    def init_run(self, config: Dict[str, Any], run_name: str = wandb_config.run_name, 
                 tags: Optional[List[str]] = None) -> bool:
        """
        Initialize a new wandb run.
        
        Args:
            config: Configuration dictionary to log
            run_name: Optional custom run name
            tags: Optional list of tags (will be merged with default tags)
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Merge tags
        all_tags = wandb_config.tags.copy()
        if tags:
            all_tags.extend(tags)
        
        # Validate API key and login
        api_key = self._get_api_key()
        if not api_key:
            self.enabled = False
            return False

        wandb.login(key=api_key)

        run_params = {
            "project": wandb_config.project,
            "entity": wandb_config.team,
            "name": run_name or wandb_config.run_name,
            "tags": all_tags,
            "config": config,
            "reinit": True  # Allow multiple runs in same process
        }
        # Initialize wandb run
        self.run = wandb.init(**run_params)
        
        self.initialized = True

        logger.info(f"✅ Wandb run initialized: {self.run.name}")
        logger.info(f"   Project: {wandb_config.project}")
        logger.info(f"   Entity: {wandb_config.entity}")
        logger.info(f"   Tags: {all_tags}")
        
        return True
    

    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log summary metrics to wandb.
        
        Args:
            summary: Dictionary of summary metrics to log
        """
        self.run.summary.update(summary)
        logger.info(f"Logged summary metrics: {list(summary.keys())}")

    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        self.run.log(metrics, step=step)
    
    @requires_wandb_initialized
    @safe_wandb_operation()
    def finish(self) -> None:
        """Finish the current wandb run."""
        self.run.finish()
        logger.info("Wandb run finished")
        self.initialized = False

    @requires_wandb_initialized
    @safe_wandb_operation()
    def watch_model(self, model: torch.nn.Module, log_freq: Optional[int] = None) -> None:
        """
        Watch model for gradients and parameters.
        
        Args:
            model: PyTorch model to watch
            log_freq: Frequency to log gradients
        """
        freq = log_freq or wandb_config.log_frequency
        self.run.watch(model, log_freq=freq)
        logger.info(f"Started watching model with log frequency: {freq}")

    def _get_checkpoint_name(self) -> str:
        """
        Get the checkpoint name based on the current run name.
        
        Returns:
            Checkpoint name in format: {run_name}_checkpoint
        """
        if self.run and self.run.name:
            return f"{self.run.name}_checkpoint"
        else:
            # Fallback to default if run not initialized
            return "checkpoint"

    def get_checkpoint_name(self) -> str:
        """
        Public method to get the current checkpoint name.
        
        Returns:
            Checkpoint name in format: {run_name}_checkpoint
        """
        return self._get_checkpoint_name()

    @requires_wandb_enabled
    @safe_wandb_operation()
    def load_model_from_registry(self, model_class, collection_name: str, 
                                alias: str = "latest", device: str = "cpu", **kwargs) -> Optional[torch.nn.Module]:
        """
        Load model from wandb model registry.
        
        Args:
            model_class: The model class to instantiate
            collection_name: Name of the model collection in wandb registry
            alias: Model version alias (e.g., "latest", "best", "v1")
            device: Device to load the model on
            
        Returns:
            Loaded model instance or None if loading failed
        """
        api = self._login_and_get_api()
        if not api:
            return None
        
        # Download artifact
        artifact_path = self._construct_artifact_path(collection_name, alias)
        model_artifact = api.artifact(artifact_path, type="model")
        model_dir = model_artifact.download()

        model_file_name = None
        for file in model_artifact.files():
            if file.name.endswith('.pth'):
                model_file_name = file.name
                break

        if not model_file_name:
            logger.error(f"No .pth file found in artifact: {artifact_path}")
            return None

        model_path = os.path.join(model_dir, model_file_name)
        logger.info(f"Model path: {model_path}")
        
        if os.path.exists(model_path):
            # Initialize model and load weights
            model = model_class(**kwargs)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            logger.info(f"✓ Successfully loaded model from wandb registry: {collection_name}:{alias}")
            return model
        else:
            logger.warning(f"Model file not found in wandb artifact: {model_path}")
            return None
        
    @requires_wandb_initialized
    @safe_wandb_operation()
    def save_model_to_registry(self, model: torch.nn.Module, collection_name: str, 
                            alias: str = "latest", file_name: str = "model.pth", 
                            metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> None:
        """
        Save model to wandb model registry with automatic version management.
        Keeps only the last versions plus any versions tagged.

        Args:
            model: PyTorch model to save
            collection_name: Name for the model collection in wandb registry
            alias: Model version alias (e.g., "latest", "best", "v1")
            file_name: Name of the model file to save
            metadata: Additional metadata to include
            tags: Optional list of tags to add to the artifact
        """
        # Create artifact
        artifact = wandb.Artifact(
            name=collection_name,
            type="model",
            metadata=metadata or {}
        )

        # Save model state dict
        model_path = file_name
        torch.save(model.state_dict(), model_path)
        artifact.add_file(model_path)
        
        # Log artifact
        logged_artifact = self.run.log_artifact(artifact, type="model", tags=tags)
        target_path = f"wandb-registry-model/{collection_name}"
        self.run.link_artifact(logged_artifact, target_path=target_path)

        logger.info(f"✓ Model saved to wandb registry: {collection_name}")

        # Clean up old model versions
        self._cleanup_old_model_versions(collection_name)
        
        # Clean up old checkpoints since final model is now saved
        checkpoint_name = self._get_checkpoint_name()
        self._cleanup_old_checkpoints(checkpoint_name, keep_latest=0)  # Remove all checkpoints

    def _cleanup_old_model_versions(self, collection_name: str, keep_latest: int = 3) -> None:
        """
        Clean up old model versions, keeping only the latest N versions and any marked "do not delete".
        
        Args:
            collection_name: Name of the model artifact
            keep_latest: Number of latest versions to keep (default: 3)
        """
        api = self._login_and_get_api()
        if not api:
            return
        
        # Get all versions of this model
        artifact_collection_name = f"{wandb_config.team}/{wandb_config.project}/{collection_name}"

        try:
            # List all versions of the artifact
            versions = list(api.artifact_type("model", project=wandb_config.project).collection(collection_name).artifacts())
            
            if len(versions) <= keep_latest:
                logger.info(f"Only {len(versions)} versions exist, no cleanup needed")
                return
            
            # Sort versions by creation time (newest first)
            versions.sort(key=lambda x: x.created_at, reverse=True)
            
            # Identify versions to delete
            versions_to_delete = []
            for i, version in enumerate(versions):
                # Keep the latest N versions
                if i < keep_latest:
                    continue
                
                # Check if this version has "do not delete" alias or tag
                do_not_delete = False
                
                # Check aliases
                for alias in version.aliases:
                    if alias in wandb_config.model_tags_to_skip_deletion:
                        do_not_delete = True
                        break
                
                # Check metadata for do not delete flag
                if not do_not_delete and version.metadata:
                    if version.metadata.get("do_not_delete", False) or \
                       "do not delete" in str(version.metadata.get("tags", "")).lower():
                        do_not_delete = True
                
                if not do_not_delete:
                    versions_to_delete.append(version)
            
            # Delete old versions
            for version in versions_to_delete:
                try:
                    version.delete()
                    logger.info(f"Deleted old model version: {collection_name}:v{version.version}")
                except Exception as e:
                    logger.warning(f"Failed to delete version {version.version}: {e}")
            
            if versions_to_delete:
                logger.info(f"Cleaned up {len(versions_to_delete)} old model versions")
            else:
                logger.info("No old versions to clean up (all marked as 'do not delete')")
                
        except Exception as e:
            logger.warning(f"Failed to access artifact collection {artifact_collection_name}: {e}")

    @requires_wandb_initialized
    @safe_wandb_operation()
    def update_run_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update the wandb run config and tags with the contents of a dictionary and a list of tags.
        Args:
            config_dict: Dictionary of config values to update in the wandb run
        """
        self.run.config.update(config_dict, allow_val_change=True)
        logger.debug(f"Updated wandb run config with: {list(config_dict.keys())}")


    @requires_wandb_initialized
    @safe_wandb_operation()
    def save_checkpoint(self, epoch: int, model_state_dict: Dict[str, Any], 
                   optimizer_state_dict: Dict[str, Any], loss: float, 
                   model_name: str, model_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Save model checkpoint to wandb and clean up previous versions.
        
        Args:
            epoch: Current epoch number
            model_state_dict: Model state dictionary
            optimizer_state_dict: Optimizer state dictionary  
            loss: Current loss value
            model_name: Name of the model for artifact naming
            model_config: Optional model configuration metadata
            
        Returns:
            Path to saved checkpoint artifact, or None if failed
        """
        # Create temporary checkpoint file
        with tempfile.NamedTemporaryFile(suffix=f'_epoch_{epoch}.pth', delete=False) as tmp_file:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'model_config': model_config or {}
            }
            
            torch.save(checkpoint_data, tmp_file.name)
            checkpoint_path = tmp_file.name
        
        # Create wandb artifact
        artifact_name = self._get_checkpoint_name()
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model_checkpoint",
            description=f"Model checkpoint for epoch {epoch}",
            metadata={
                "epoch": epoch,
                "loss": loss,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Add checkpoint file to artifact
        artifact.add_file(checkpoint_path, name=f"checkpoint_epoch_{epoch}.pth")
        
        # Log artifact to wandb (this automatically versions it)
        self.run.log_artifact(artifact)
        
        # Clean up temporary file
        os.unlink(checkpoint_path)
        
        logger.info(f"Saved model checkpoint to wandb for epoch {epoch}")
        
        # Clean up previous checkpoint artifacts (keep only latest)
        self._cleanup_old_checkpoints(artifact_name)

        return f"{artifact_name}:latest"


    def _cleanup_old_checkpoints(self, artifact_name: str, keep_latest: int = 1):
        """
        Clean up old wandb checkpoint artifacts, keeping only the latest N versions.
        
        Args:
            artifact_name: Name of the artifact to clean up
            keep_latest: Number of latest versions to keep (default: 1)
        """
        api = self._login_and_get_api()
        if not api or not self.run:
            return
            
        # List all versions of this artifact
        try:
            artifact_versions = list(api.artifact_versions(
                "model_checkpoint", 
                f"{self.run.entity}/{self.run.project}/{artifact_name}"
            ))
            
            # Sort by version (latest first) and skip the ones we want to keep
            if len(artifact_versions) <= keep_latest:
                return
                
            versions_to_delete = artifact_versions[keep_latest:]
            
            for version in versions_to_delete:
                try:
                    version.delete()
                    logger.info(f"Deleted old checkpoint artifact: {version.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete old checkpoint {version.name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to access checkpoint artifacts: {e}")

    @requires_wandb_enabled
    @safe_wandb_operation()
    def load_checkpoint(self, artifact_name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Load model checkpoint from wandb.
        
        Args:
            artifact_name: Name of the wandb artifact
            version: Version of the artifact to load (default: "latest")
            
        Returns:
            Checkpoint data dictionary or None if failed
        """
        api = self._login_and_get_api()
        if not api:
            return None
        
        # Download artifact
        artifact_path = self._construct_artifact_path(artifact_name, version)
        try:
            artifact = api.artifact(artifact_path, type="model_checkpoint")
            artifact_dir = artifact.download()
        except Exception as e:
            logger.error(f"Failed to download artifact {artifact_name}:{version} from wandb: {e}")
            return None

        # Find checkpoint file
        checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
        
        if not checkpoint_files:
            logger.error(f"No checkpoint files found in artifact {artifact_name}:{version}")
            return None
        
        # Load the checkpoint (assume first .pth file)
        checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        logger.info(f"Loaded checkpoint from wandb: {artifact_name}:{version} (epoch {checkpoint_data.get('epoch', 'unknown')})")
        return checkpoint_data


    def resume_training_from_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                                  artifact_name: str, version: str = "latest") -> int:
        """
        Resume training from a wandb checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into  
            artifact_name: Name of the wandb checkpoint artifact
            version: Version to load (default: "latest")
            
        Returns:
            Starting epoch number for resuming training
        """
        checkpoint_data = self.load_checkpoint(artifact_name, version)
        
        if checkpoint_data is None:
            logger.warning("Could not load checkpoint, starting training from epoch 1")
            return 1
        
        try:
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Get starting epoch
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            
            logger.info(f"Resumed training from epoch {start_epoch}")
            return start_epoch
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return 1


    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_training_metrics(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, 
                        learning_rate: Optional[float] = None, **kwargs) -> None:
        """
        Log training metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for the epoch
            val_loss: Optional validation loss
            learning_rate: Current learning rate
            **kwargs: Additional metrics to log
        """
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            **kwargs
        }
        
        if val_loss is not None:
            metrics["val_loss"] = val_loss
            
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate
        
        self.run.log(metrics)
        logger.debug(f"Logged training metrics for epoch {epoch}")

    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_batch_metrics(self, batch_idx: int, epoch: int, batch_loss: float, **kwargs) -> None:
        """
        Log batch-level metrics during training.
        
        Args:
            batch_idx: Current batch index
            epoch: Current epoch number
            batch_loss: Loss for this batch
            **kwargs: Additional batch metrics to log
        """
        metrics = {
            "batch_loss": batch_loss,
            "epoch": epoch,
            "batch": batch_idx,
            **kwargs
        }
        
        self.run.log(metrics)

wandb_logger = WandbLogger()
            