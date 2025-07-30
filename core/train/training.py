import os
import torch
import logging
from typing import Optional, Any, Dict
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import torch.nn as nn
import wandb
from core.config.all_config import model_config, training_config, wandb_config
from .wandb_logger import wandb_logger

logger = logging.getLogger(__name__)


class Training:
    """
    A training class specifically for neural network training using triplet loss.
    Focused on training lacrosse player re-identification models.
    """
    
    def __init__(self, 
                 train_dir: str,
                 storage_client: Any,
                 embedding_dim: Optional[int],
                 dropout_rate: Optional[float],
                 margin: float = training_config.margin,
                 learning_rate=training_config.learning_rate,
                 batch_size=training_config.batch_size,
                 num_epochs=training_config.num_epochs,
                 weight_decay=training_config.weight_decay,
                 scheduler_patience = training_config.lr_scheduler_patience,
                 scheduler_threshold = training_config.lr_scheduler_threshold,
                 lr_scheduler_min_lr = training_config.lr_scheduler_min_lr,
                 lr_scheduler_factor = training_config.lr_scheduler_factor,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
                 ):
        """
        Initialize the training class with hyperparameters.
        
        Args:
            train_dir: Directory containing training data (local or GCS blob prefix)
            storage_client: Google Storage client for GCS operations (required for GCS paths)
            embedding_dim: Dimension of the embedding vector
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            margin: Margin for triplet loss
            device: Device to run the model on (CPU, GPU, or MPS)
        """
        self.train_dir = train_dir
        self.storage_client = storage_client
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.margin = margin
        self.weight_decay = weight_decay
        self.device = device
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.lr_scheduler_min_lr = lr_scheduler_min_lr
        self.lr_scheduler_factor = lr_scheduler_factor

        self.model = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.dataloader = None
        load_dotenv()

    def load_model_from_wandb(self, model_class, model_name: str, alias: Optional[str]):
        """
        Load model from wandb model registry.
        
        Args:
            model_class: The model class to instantiate
            model_name: Name of the model in wandb registry
            alias: Model version alias (e.g., "latest", "best", "v1")
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Use the centralized wandb model loading

            registry_kwargs = {
                'model_class': lambda: model_class(),
                'collection_name': model_name,
                'device': str(self.device)
            }
            if alias is not None:
                registry_kwargs['alias'] = alias
            loaded_model = wandb_logger.load_model_from_registry(**registry_kwargs)
            
            if loaded_model is not None:
                self.model = loaded_model
                logger.info(f"✓ Successfully loaded model from wandb registry: {model_name}:{alias}")
                return True
            else:
                logger.info(f"Could not load model from wandb registry")
                return False
                
        except Exception as e:
            logger.info(f"Could not load model from wandb registry: {e}")
            return False

    def save_model(self, model_name: str):
        """
        Save the trained model weights both locally and to wandb registry.
        
        Args:
            model_name: Name of the model to save in wandb registry
            
        Raises:
            RuntimeError: If no model to save
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        
        try:
            # Collect additional metadata for reproducibility and traceability
            import datetime
            import subprocess
            metadata = {
                "embedding_dim": self.embedding_dim,
                "dropout_rate": self.dropout_rate,
                "device": str(self.device),
                "num_epochs": self.num_epochs,
                "margin": self.margin,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "scheduler_type": "ReduceLROnPlateau",
                "scheduler_patience": self.scheduler_patience,
                "scheduler_threshold": self.scheduler_threshold,
                "lr_scheduler_min_lr": self.lr_scheduler_min_lr,
                "lr_scheduler_factor": self.lr_scheduler_factor,
                "optimizer_type": "Adam",
                "train_dir": self.train_dir,
                "model_architecture": type(self.model).__name__ if self.model is not None else None,
                "timestamp": datetime.datetime.now().isoformat(),
                "wandb_run_id": wandb.run.id if hasattr(wandb, 'run') and wandb.run is not None else None,
                "notes": getattr(self, 'notes', None),
            }
            # Try to get git commit hash if available
            try:
                git_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
                metadata["git_commit_hash"] = git_commit_hash
            except Exception:
                metadata["git_commit_hash"] = None

            # Optionally add dataset info if available
            if hasattr(self, 'dataloader') and self.dataloader is not None:
                try:
                    metadata["train_dataset_size"] = len(self.dataloader.dataset)
                except Exception:
                    metadata["train_dataset_size"] = None
            if hasattr(self, 'dataloader') and self.dataloader is not None:
                try:
                    metadata["train_num_batches"] = len(self.dataloader)
                except Exception:
                    metadata["train_num_batches"] = None

            wandb_logger.save_model_to_registry(
                model=self.model,
                collection_name=model_name,
                metadata=metadata
            )
            logger.info(f"✓ Model saved to wandb registry: {model_name}:latest")
        except Exception as e:
            logger.error(f"Failed to save model to wandb registry: {e}")

    def setup_data(self, dataset_class, transform=None):
        """
        Setup the dataset and dataloader for training.
        
        Args:
            dataset_class: The dataset class to use (e.g., LacrossePlayerDataset)
            transform: Data transforms to apply
            
        Raises:
            FileNotFoundError: If training directory doesn't exist
            ValueError: If insufficient player folders for training
        """
        # Check if we're working with GCS or local filesystem
        is_gcs_path = self.storage_client is not None
        
        if is_gcs_path:
            # For GCS paths, validate by listing blobs with the prefix
            try:
                all_blobs = list(self.storage_client.list_blobs(prefix=self.train_dir))
                if not all_blobs:
                    raise FileNotFoundError(f"Training directory does not exist in GCS: {self.train_dir}")
                
                # Check for player folders by looking for blobs with player subfolders
                player_folders = set()
                for blob in all_blobs:
                    # Remove the train_dir prefix and look for player folders
                    relative_path = blob[len(self.train_dir):].lstrip('/')
                    if '/' in relative_path:
                        player = relative_path.split('/')[0]
                        if player and blob.lower().endswith(('.jpg', '.png', '.jpeg')):
                            player_folders.add(player)
                
                if len(player_folders) < 2:
                    raise ValueError(f"Need at least 2 player folders for triplet loss training! Found: {len(player_folders)}")
                
                logger.info(f"Found {len(player_folders)} player folders in GCS path: {self.train_dir}")
                
            except Exception as e:
                if isinstance(e, (FileNotFoundError, ValueError)):
                    raise
                raise FileNotFoundError(f"Failed to access GCS training directory: {self.train_dir} - {str(e)}")
        else:
            # For local filesystem paths, use original validation
            if not os.path.exists(self.train_dir):
                raise FileNotFoundError(f"Training directory does not exist: {self.train_dir}")
            
            train_folders = [d for d in os.listdir(self.train_dir) 
                            if os.path.isdir(os.path.join(self.train_dir, d))]
            
            if len(train_folders) < 2:
                raise ValueError("Need at least 2 player folders for triplet loss training!")
            
            logger.info(f"Found {len(train_folders)} player folders in {self.train_dir}")
        
        # Setup dataset and dataloader
        if is_gcs_path:
            # For GCS, pass storage_client to dataset
            if transform is not None:
                train_dataset = dataset_class(image_dir=self.train_dir, storage_client=self.storage_client, transform=transform)
            else:
                train_dataset = dataset_class(image_dir=self.train_dir, storage_client=self.storage_client)
        else:
            # For local filesystem, use original approach
            if transform is not None:
                train_dataset = dataset_class(image_dir=self.train_dir, transform=transform)
            else:
                train_dataset = dataset_class(image_dir=self.train_dir)

        n_workers = getattr(training_config, 'num_workers', 0)
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers
        )
        
        logger.info(f"Dataset setup complete:")
        logger.info(f"  Dataset size: {len(train_dataset)} images")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Number of workers: {n_workers}")
        logger.info(f"  Number of batches: {len(self.dataloader)}")

    def setup_model(self, model_class, model_name: str, force_pretrained: bool = False):
    
        """
        Setup the model, loss function, and optimizer for training.
        First attempts to load from wandb registry, then falls back to local weights.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
            model_name: The name of the model (used for logging and saving)
            force_pretrained: If True, ignore saved weights and start with pre-trained backbone
            
        Raises:
            RuntimeError: If model setup fails
        """
        try:
            model_loaded = False
            # Prepare kwargs for model initialization
            model_kwargs = {}
            if self.embedding_dim is not None:
                model_kwargs['embedding_dim'] = self.embedding_dim
            if self.dropout_rate is not None:
                model_kwargs['dropout_rate'] = self.dropout_rate

            # Try to load from wandb registry first (unless forcing pretrained)
            if force_pretrained:
                logger.info("Forcing fresh start with pre-trained ResNet18 weights")
                self.model = model_class(**model_kwargs)
            else:
                model_loaded = self.load_model_from_wandb(model_class, model_name=model_name, alias="latest", **model_kwargs)
            if not model_loaded:
                logger.info("No wandb model found, will use local weights or pre-trained backbone")
                self.model = model_class(**model_kwargs)

            self.model.to(self.device)
            # Setup training components
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            # Scheduler uses its own patience/threshold for LR adjustment only
        
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold,
                min_lr=self.lr_scheduler_min_lr
            )

            logger.info(f"Training model initialized with device: {self.device}")
            logger.info(f"Training directory: {self.train_dir}")
            logger.info(f"Batch size: {self.batch_size}")
            logger.info(f"Number of epochs: {self.num_epochs}")
            logger.info(f"Triplet margin: {self.margin}")
            logger.info(f"Optimizer: Adam (initial lr={self.learning_rate}, weight_decay={self.weight_decay})")
            logger.info(f"LR Scheduler: ReduceLROnPlateau (patience={self.scheduler_patience}, threshold={self.scheduler_threshold})")
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            raise RuntimeError(f"Failed to setup model: {e}")

    def train(self, margin_decay_rate: float = training_config.margin_decay_rate, margin_change_threshold: float = training_config.margin_change_threshold):
        """
        Execute the main training loop with early stopping.
        
        Returns:
            The trained model
            
        Raises:
            RuntimeError: If required components are not setup
        """
        if self.model is None or self.dataloader is None:
            raise RuntimeError("Model and dataloader must be setup before training")
        
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Optimizer and loss function must be setup before training")
        
        logger.info(f"Starting training for {self.num_epochs} epochs")

        # Log dataset info to wandb
        if hasattr(self, 'dataset') and wandb_config.enabled:
            dataset_size = len(self.dataset)
            num_players = len(self.dataset.players)
            player_stats = {player: len(self.dataset.player_to_images[player]) 
                           for player in self.dataset.players}
            
            wandb_logger.log_dataset_info(
                dataset_path=self.train_dir,
                dataset_size=dataset_size,
                num_players=num_players,
                player_stats=player_stats
            )
            

        # Early stopping configuration
        early_stopping_threshold = training_config.early_stopping_loss_ratio * self.margin
        early_stopping_patience = getattr(training_config, 'early_stopping_patience', None)
        
        logger.info(f"Early stopping threshold set to {early_stopping_threshold:.4f}")
        if early_stopping_patience is not None:
            logger.info(f"Early stopping patience set to {early_stopping_patience} epochs")
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
        current_margin = self.margin
        self.loss_fn = nn.TripletMarginLoss(margin=current_margin, p=2)
        for epoch in range(self.num_epochs):
            # Calculate new margin for this epoch
            new_margin = self.margin * (margin_decay_rate ** epoch)
            if abs(new_margin - current_margin) > margin_change_threshold:
                self.margin = new_margin
                self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)
                logger.info(f"Margin updated for epoch {epoch+1}: {new_margin:.4f}")
            else:
                logger.debug(f"Margin unchanged for epoch {epoch+1}: {current_margin:.4f}")

            self.model.train()
            running_loss = 0.0
            batch_count = 0

            for i, (anchor, positive, negative, _) in enumerate(self.dataloader):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                emb_anchor, emb_positive, emb_negative = self.model.forward_triplet(
                    anchor, positive, negative)

                # Calculate loss
                loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                batch_count += 1

            # Calculate and log epoch summary
            epoch_loss = running_loss / batch_count if batch_count > 0 else 0.0
            logger.info(f"=== Epoch {epoch+1}/{self.num_epochs} Summary ===")
            logger.info(f"Average Loss: {epoch_loss:.4f}")
            logger.info(f"Margin used: {current_margin:.4f}")

            # Log epoch metrics to wandb
            if wandb_config.enabled:
                # Use a step that is guaranteed to be after all batch steps for this epoch
                epoch_end_step = (epoch + 1) * len(self.dataloader)
                wandb_logger.log_metrics({
                    "epoch_loss": epoch_loss,
                    "epoch": epoch + 1,
                    "margin": current_margin
                }, step=epoch_end_step)

            # Early stopping based on loss threshold
            if epoch_loss < early_stopping_threshold:
                logger.info(f"Early stopping triggered (loss {epoch_loss:.4f} < threshold {early_stopping_threshold:.4f})")
                break

            # Early stopping based on patience
            if early_stopping_patience is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    logger.debug(f"New best loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.debug(f"Patience counter: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered due to patience ({patience_counter} epochs without improvement)")
                    break

            # Step the learning rate scheduler
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate after scheduler step: {current_lr:.6f}")
        logger.info("Training completed successfully")
        return self.model

    def train_and_save(self, model_class, dataset_class, model_name: str, transform: Optional[Any] = None, force_pretrained: bool = False):
        """
        Complete training pipeline: setup data, setup model, train, and save.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
            dataset_class: The dataset class to use (e.g., LacrossePlayerDataset)
            model_name: Name of the model to save in wandb registry
            transform: Data transforms to apply
            force_pretrained: If True, ignore saved weights and start fresh
            
        Returns:
            The trained model
            
        Raises:
            Exception: If any step in the training pipeline fails
        """
        try:
            logger.info("Starting complete training pipeline")
            
            # Setup data
            logger.info("Setting up training data...")
            self.setup_data(dataset_class, transform)
            
            # Setup model
            logger.info("Setting up model...")
            self.setup_model(model_class, model_name=model_name, force_pretrained=force_pretrained)

            # Train
            logger.info("Starting training...")
            trained_model = self.train()
            
            # Save
            logger.info("Saving model...")
            self.save_model(model_name=model_name)
              
            logger.info("Training pipeline completed successfully")
            
            return trained_model
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

    def get_training_info(self):
        """
        Get information about the current training setup.
        
        Returns:
            Dictionary with training configuration details
        """
        return {
            'train_dir': self.train_dir,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'margin': self.margin,
            'weight_decay': self.weight_decay,
            'scheduler_patience': self.scheduler_patience,
            'scheduler_threshold': self.scheduler_threshold,
            'lr_scheduler_min_lr': self.lr_scheduler_min_lr,
            'lr_scheduler_factor': self.lr_scheduler_factor,
            'device': str(self.device),
            'model_loaded': self.model is not None,
            'dataloader_ready': self.dataloader is not None,
            'optimizer_ready': self.optimizer is not None,
            'loss_fn_ready': self.loss_fn is not None
        }
