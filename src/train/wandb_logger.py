import os
import logging
from typing import Dict, Any, Optional, List
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
        if not self.enabled:
            return False
            
        try:
            # Merge tags
            all_tags = wandb_config.tags.copy()
            if tags:
                all_tags.extend(tags)
            
            # Validate API key and login
            api_key = os.environ.get("WANDB_API_KEY")
            if not api_key:
                logger.error("WANDB_API_KEY environment variable not found")
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
            
        except Exception as e:
            logger.error(f"Failed to initialize wandb run: {e}")
            self.enabled = False
            return False
    

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log summary metrics to wandb.
        
        Args:
            summary: Dictionary of summary metrics to log
        """
        if not self.enabled or not self.initialized:
            return
            
        try:
            if self.run is None:
                raise RuntimeError("Wandb run is not initialized. Call init_run() first.")
                
            self.run.summary.update(summary)
            logger.info(f"Logged summary metrics: {list(summary.keys())}")

        except Exception as e:
            logger.warning(f"Failed to log summary metrics to wandb: {e}")


    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled or not self.initialized or self.run is None:
            logger.warning("Wandb logging is not enabled or initialized.")
            return
            
        try:
            self.run.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")
    
   

    def finish(self) -> None:
        """Finish the current wandb run."""
        if not self.enabled or not self.initialized or self.run is None:
            logger.warning("Wandb logging is not enabled or initialized.")
            return
            
        try:
            self.run.finish()
            logger.info("Wandb run finished")
            self.initialized = False
            
        except Exception as e:
            logger.warning(f"Failed to finish wandb run: {e}")

    def watch_model(self, model: torch.nn.Module, log_freq: Optional[int] = None) -> None:
        """
        Watch model for gradients and parameters.
        
        Args:
            model: PyTorch model to watch
            log_freq: Frequency to log gradients
        """
        if not self.enabled or not self.initialized or self.run is None:
            logger.warning("Wandb logging is not enabled or initialized.")
            return
            
        try:
            freq = log_freq or wandb_config.log_frequency
            self.run.watch(model, log_freq=freq)
            logger.info(f"Started watching model with log frequency: {freq}")
            
        except Exception as e:
            logger.warning(f"Failed to watch model with wandb: {e}")

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
        if not self.enabled:
            logger.info("Wandb not enabled, skipping wandb model loading")
            return None
            
        try:
            # Use wandb.Api to avoid creating a run for just downloading
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if not wandb_api_key:
                logger.warning("WANDB_API_KEY not found, cannot download model from registry.")
                return None
            
            wandb.login(key=wandb_api_key)
            api = wandb.Api()
            
            # The artifact path is constructed from entity/project/artifact_name:alias
            artifact_path = f"{wandb_config.team}/{wandb_config.project}/{collection_name}:{alias}"
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
                
        except Exception as e:
            logger.info(f"Could not load model from wandb registry: {e}")
            return None
        
    def save_model_to_registry(self, model: torch.nn.Module, collection_name: str, 
                            alias: str = "latest", file_name: str = "model.pth", metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> None:
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
        if not self.enabled or not self.initialized or self.run is None:
            return
            
        try:
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

            # Clean up old versions
            self._cleanup_old_model_versions(collection_name)

        except Exception as e:
            logger.warning(f"Failed to save model to wandb registry: {e}")

    def _cleanup_old_model_versions(self, collection_name: str, keep_latest: int = 3) -> None:
        """
        Clean up old model versions, keeping only the latest N versions and any marked "do not delete".
        
        Args:
            collection_name: Name of the model artifact
            keep_latest: Number of latest versions to keep (default: 3)
        """
        try:
            # Get wandb API
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if not wandb_api_key:
                logger.warning("WANDB_API_KEY not found, cannot cleanup old versions.")
                return
                
            wandb.login(key=wandb_api_key)
            api = wandb.Api()
            
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
                
        except Exception as e:
            logger.warning(f"Failed to cleanup old model versions: {e}")

    def update_run_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update the wandb run config and tags with the contents of a dictionary and a list of tags.
        Args:
            config_dict: Dictionary of config values to update in the wandb run
            tags: Optional list of tags to update in the wandb run
        """
        if not self.enabled or not self.initialized or self.run is None:
            return
        try:
            self.run.config.update(config_dict, allow_val_change=True)
            logger.debug(f"Updated wandb run config with: {list(config_dict.keys())}")
        except Exception as e:
            logger.warning(f"Failed to update wandb run config or tags: {e}")

# Global wandb logger instance
wandb_logger = WandbLogger()
