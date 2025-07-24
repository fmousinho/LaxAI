import os
import logging
from typing import Dict, Any, Optional, List
import torch
import numpy as np
from PIL import Image

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
    
    def __init__(self, enabled: Optional[bool] = None):
        """
        Initialize the wandb logger.
        
        Args:
            enabled: Override config setting for wandb logging
        """
        self.enabled = enabled if enabled is not None else wandb_config.enabled
        self.run = None
        self.initialized = False
        
        if not WANDB_AVAILABLE:
            self.enabled = False
            logger.warning("wandb package not available, disabling wandb logging")
    
    def init_run(self, config: Dict[str, Any], run_name: Optional[str] = None, 
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
            
            # Initialize wandb run
            self.run = wandb.init(
                project=wandb_config.project,
                entity=wandb_config.entity,
                name=run_name,
                tags=all_tags,
                config=config,
                reinit=True  # Allow multiple runs in same process
            )
            
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
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled or not self.initialized:
            return
            
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")
    
    def log_dataset_info(self, dataset_path: str, dataset_size: int, 
                        num_players: int, player_stats: Dict[str, int]) -> None:
        """
        Log dataset information to wandb.
        
        Args:
            dataset_path: Path to the dataset
            dataset_size: Total number of images
            num_players: Number of unique players
            player_stats: Dictionary mapping player names to image counts
        """
        if not self.enabled or not self.initialized:
            return
            
        try:
            # Log dataset summary
            dataset_info = {
                "dataset/path": dataset_path,
                "dataset/total_images": dataset_size,
                "dataset/num_players": num_players,
                "dataset/avg_images_per_player": dataset_size / num_players if num_players > 0 else 0,
                "dataset/min_images_per_player": min(player_stats.values()) if player_stats else 0,
                "dataset/max_images_per_player": max(player_stats.values()) if player_stats else 0
            }
            
            # Add individual player stats
            for player, count in player_stats.items():
                dataset_info[f"dataset/player_{player}_images"] = count
            
            wandb.log(dataset_info)
            logger.info(f"Logged dataset info to wandb: {dataset_size} images, {num_players} players")
            
        except Exception as e:
            logger.warning(f"Failed to log dataset info to wandb: {e}")
    
    def log_sample_images(self, dataset, max_images_per_player: Optional[int] = None) -> None:
        """
        Log sample training images to wandb.
        
        Args:
            dataset: Training dataset object
            max_images_per_player: Maximum images to log per player (None = log all if config allows)
        """
        if not self.enabled or not self.initialized:
            return
            
        if not wandb_config.log_sample_images:
            return
            
        try:
            images_to_log = []
            
            # Determine how many images to log per player
            if wandb_config.log_all_images:
                images_per_player = None  # Log all images
            else:
                images_per_player = max_images_per_player or wandb_config.sample_images_count
            
            # Collect sample images from each player
            for player in dataset.players:
                player_images = dataset.player_to_images[player]
                
                # Select images to log
                if images_per_player is None:
                    selected_images = player_images  # All images
                else:
                    selected_images = player_images[:images_per_player]
                
                for img_path in selected_images:
                    try:
                        # Load and convert image
                        img = Image.open(img_path).convert('RGB')
                        
                        # Create wandb image with caption
                        wandb_image = wandb.Image(
                            img, 
                            caption=f"Player: {player} | Path: {os.path.basename(img_path)}"
                        )
                        images_to_log.append(wandb_image)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path}: {e}")
            
            # Log all images at once
            if images_to_log:
                wandb.log({"training_samples": images_to_log})
                logger.info(f"Logged {len(images_to_log)} sample images to wandb")
            
        except Exception as e:
            logger.warning(f"Failed to log sample images to wandb: {e}")
    
    def log_model_artifact(self, model_path: str, model_name: Optional[str] = None, 
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model as wandb artifact.
        
        Args:
            model_path: Path to the saved model file
            model_name: Name for the model artifact
            metadata: Additional metadata to include
        """
        if not self.enabled or not self.initialized:
            return
            
        if not wandb_config.save_model_artifacts:
            return
            
        try:
            artifact_name = model_name or wandb_config.model_name
            
            # Create artifact
            model_artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata=metadata or {}
            )
            
            # Add model file
            model_artifact.add_file(model_path)
            
            # Log artifact
            wandb.log_artifact(model_artifact)
            logger.info(f"Logged model artifact to wandb: {artifact_name}")
            
        except Exception as e:
            logger.warning(f"Failed to log model artifact to wandb: {e}")
    
    def finish(self) -> None:
        """Finish the current wandb run."""
        if not self.enabled or not self.initialized:
            return
            
        try:
            wandb.finish()
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
        if not self.enabled or not self.initialized:
            return
            
        try:
            freq = log_freq or wandb_config.log_frequency
            wandb.watch(model, log_freq=freq)
            logger.info(f"Started watching model with log frequency: {freq}")
            
        except Exception as e:
            logger.warning(f"Failed to watch model with wandb: {e}")


# Global wandb logger instance
wandb_logger = WandbLogger()
