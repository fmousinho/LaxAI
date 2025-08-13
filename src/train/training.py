import os
import torch
import logging
from typing import Optional, Any, Dict
from torch.utils.data import DataLoader, Dataset
from utils.env_or_colab import load_env_or_colab
import torch.nn as nn
import numpy as np
import wandb
from config.all_config import model_config, training_config, wandb_config
from .wandb_logger import wandb_logger

logger = logging.getLogger(__name__)


class Training:
    """
    A training class specifically for neural network training using triplet loss.
    Focused on training lacrosse player re-identification models.

    """
    
    def __init__(self, 
                device: Any = None,
                **kwargs):
        """
        Initialize the training class with hyperparameters.
        
        Args:
            device: Device to run the model on (CPU, GPU, or MPS)
            All other hyperparameters must be provided as kwargs or present in training_config.

        Kwargs (defaults to training_config if not provided):
            margin (float): Margin for triplet loss
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            num_epochs (int): Number of training epochs
            weight_decay (float): Weight decay for optimizer
            lr_scheduler_patience (int): Patience for LR scheduler
            lr_scheduler_threshold (float): Threshold for LR scheduler
            lr_scheduler_min_lr (float): Minimum LR for scheduler
            lr_scheduler_factor (float): Factor for LR scheduler
            force_pretraining (bool): True if pretrained should use ResNet defaults
        If any required hyperparameter is missing in both kwargs and training_config, a ValueError will be raised.
        """

     
        def get_kwarg_or_config(key, config_obj=training_config, allow_none=False):
            if key in kwargs:
                return kwargs[key]
            if hasattr(config_obj, key):
                val = getattr(config_obj, key)
                if val is not None or allow_none:
                    return val
            raise ValueError(f"Missing required hyperparameter '{key}' in kwargs and config.")

        self.num_workers = get_kwarg_or_config('num_workers')
        self.margin = get_kwarg_or_config('margin')
        self.learning_rate = get_kwarg_or_config('learning_rate')
        self.batch_size = get_kwarg_or_config('batch_size')
        self.num_epochs = get_kwarg_or_config('num_epochs')
        self.weight_decay = get_kwarg_or_config('weight_decay')
        self.scheduler_patience = get_kwarg_or_config('lr_scheduler_patience')
        self.scheduler_threshold = get_kwarg_or_config('lr_scheduler_threshold')
        self.lr_scheduler_min_lr = get_kwarg_or_config('lr_scheduler_min_lr')
        self.lr_scheduler_factor = get_kwarg_or_config('lr_scheduler_factor')
        self.prefetch_factor = get_kwarg_or_config('prefetch_factor')  # Default prefetch factor if not provided
        self.force_pretraining = get_kwarg_or_config('force_pretraining')
        self.default_workers = get_kwarg_or_config('default_workers')  #used by dataloader

        # Device: direct argument, else config, else autodetect
        if device is not None:
            self.device = device
        elif getattr(training_config, 'device', None) is not None:
            self.device = getattr(training_config, 'device')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.model: nn.Module
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.dataloader = None
        self.val_dataloader = None

        EPOCHS_PER_VAL = 10
        BATCHES_PER_LOG_MSG = 10


    def _load_model_from_wandb(self, model_class, model_name: str, alias: Optional[str], **kwargs):
        """
        Load model from wandb model registry, and puts in self.model.

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
            registry_kwargs.update(kwargs)  # Merge/override with any extra kwargs
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
        Save the trained model weights to wandb registry.
        
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
                "model_architecture": type(self.model).__name__ if self.model is not None else None,
                "timestamp": datetime.datetime.now().isoformat(),
                "wandb_run_id": wandb.run.id if hasattr(wandb, 'run') and wandb.run is not None else None,
                "notes": getattr(self, 'notes', None),
            }

            # Optionally add dataset info if available
            if hasattr(self, 'dataloader') and self.dataloader is not None:
                try:
                    if hasattr(self.dataloader.dataset, "__len__") and callable(getattr(self.dataloader.dataset, "__len__", None)):
                        metadata["train_dataset_size"] = len(self.dataloader.dataset) # pyright: ignore[reportArgumentType]
                    else:
                        metadata["train_dataset_size"] = None
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


    def setup_dataloader(self, dataset, type: str = 'train'):
        """
        Setup the dataloader for the given dataset.

        Args:
            dataset: The dataset object to load
            type: Type of dataloader to create ('train' or 'val'). If 'train', uses shuffle=True.
                  If 'val', uses shuffle=False.
            
        """
        if type == 'train':
            self.dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.default_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True
            )

        elif type == 'val':
            self.val_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.default_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True
            )
        else:
            raise ValueError(f"Invalid dataloader type: {type}. Use 'train' or 'val'.")

        logger.info(f"Dataset setup complete:")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Number of workers: {self.num_workers}")
        logger.info(f"  Number of batches: {len(self.dataloader) if self.dataloader is not None else 'N/A'}")



    def setup_model(self, model_class, model_name: str, **kwargs):

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

            # Try to load from wandb registry first (unless forcing pretrained)
            if self.force_pretraining:
                logger.info("Forcing fresh start with pre-trained ResNet18 weights")
                self.model = model_class(**kwargs)
            else:
                model_loaded = self._load_model_from_wandb(model_class, model_name=model_name, alias="latest", **kwargs)
            if not model_loaded:
                logger.info("No wandb model found, will use local weights or pre-trained backbone")
                self.model = model_class(**kwargs)

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
            logger.info(f"Batch size: {self.batch_size}")
            logger.info(f"Number of epochs: {self.num_epochs}")
            logger.info(f"Triplet margin: {self.margin}")
            logger.info(f"Optimizer: Adam (initial lr={self.learning_rate}, weight_decay={self.weight_decay})")
            logger.info(f"LR Scheduler: ReduceLROnPlateau (patience={self.scheduler_patience}, threshold={self.scheduler_threshold})")
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            raise RuntimeError(f"Failed to setup model: {e}")

    def train(self, 
          margin_decay_rate: float = training_config.margin_decay_rate, 
          margin_change_threshold: float = training_config.margin_change_threshold):
        """
        Execute the main training loop with early stopping, using a validation set
        to monitor for overfitting.
        
        Args:
            margin_decay_rate: Rate at which to decay the triplet loss margin.
            margin_change_threshold: Minimum change in margin to trigger an update.
            
        Returns:
            The trained model
            
        Raises:
            RuntimeError: If required components are not setup
        """
        # ========================================================================
        # Sanity checks
        # ========================================================================
        if self.model is None or self.dataloader is None:
            raise RuntimeError("Model and dataloader must be setup before training")
        
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Optimizer and loss function must be setup before training")
        
        # ========================================================================
        # Training setup and logging
        # ========================================================================
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        # Early stopping configuration
        early_stopping_patience = getattr(training_config, 'early_stopping_patience', None)
        best_monitoring_loss = float('inf')
        patience_counter = 0

        # Margin decay setup
        current_margin = self.margin
        self.loss_fn = nn.TripletMarginLoss(margin=current_margin, p=2)
        val_dataloader = self.val_dataloader

        

        for epoch in range(self.num_epochs):
            
            # ========================================================================
            # Training Phase
            # ========================================================================
            self.model.train()
            running_loss = 0.0
            batch_count = 0

            # Update margin if decay rate is active
            new_margin = self.margin * (margin_decay_rate ** epoch)
            if abs(new_margin - current_margin) > margin_change_threshold:
                current_margin = new_margin
                self.loss_fn = nn.TripletMarginLoss(margin=current_margin, p=2)
                logger.info(f"Margin updated for epoch {epoch+1}: {current_margin:.4f}")
            else:
                logger.debug(f"Margin unchanged for epoch {epoch+1}: {current_margin:.4f}")
                
            for i, (anchor, positive, negative, _) in enumerate(self.dataloader):
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                self.optimizer.zero_grad()
                emb_anchor, emb_positive, emb_negative = self.model.forward_triplet(anchor, positive, negative)  # pyright: ignore[reportCallIssue]
                loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if (i + 1) % BATCHES_PER_LOG_MSG == 0:
                    logger.info(f"Training Batch {i+1}/{batch_count}")

                batch_count += 1

            # Calculate and log training loss
            epoch_train_loss = running_loss / batch_count if batch_count > 0 else 0.0

            # ========================================================================
            # Validation Phase (if dataloader is provided)
            # ========================================================================
            epoch_val_loss = None
            reid_metrics = {}

            if val_dataloader and (epoch + 1) % EPOCHS_PER_VAL == 0:
                self.model.eval()  # Set model to evaluation mode
                
                # 1. Calculate Validation Loss
                running_val_loss = 0.0
                val_batch_count = 0
                with torch.no_grad(): # No need to compute gradients
                    for j, (anchor, positive, negative, _) in enumerate(val_dataloader):
                        anchor = anchor.to(self.device)
                        positive = positive.to(self.device)
                        negative = negative.to(self.device)
                        
                        emb_anchor, emb_positive, emb_negative = self.model.forward_triplet(anchor, positive, negative) # pyright: ignore[reportCallIssue]
                        loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)
                        
                        running_val_loss += loss.item()

                        if (j + 1) % BATCHES_PER_LOG_MSG == 0:
                            logger.info(f"Validation Batch {j+1}/{val_batch_count}")

                        val_batch_count += 1
                
                epoch_val_loss = running_val_loss / val_batch_count if val_batch_count > 0 else 0.0
                
                # 2. Calculate Retrieval Metrics
                reid_metrics = self._evaluate_reid_metrics(val_dataloader)

            # ========================================================================
            # Log and check for early stopping
            # ========================================================================
            logger.info(f"=== Epoch {epoch+1}/{self.num_epochs} Summary ===")
            logger.info(f"Training Loss: {epoch_train_loss:.4f}")
            
            # Use validation loss for monitoring if available, otherwise fall back to training loss
            monitoring_loss = epoch_train_loss
            if epoch_val_loss is not None:
                logger.info(f"Validation Loss: {epoch_val_loss:.4f}")
                monitoring_loss = epoch_val_loss
                for key, val in reid_metrics.items():
                    logger.info(f"  - {key}: {val:.4f}")

            logger.info(f"Margin used: {current_margin:.4f}")
            
            # Log epoch metrics to wandb
            if wandb_config.enabled:
                metrics = {
                    "train_loss": epoch_train_loss,
                    "margin": current_margin,
                    "current_lr": self.optimizer.param_groups[0]['lr']
                }
                if epoch_val_loss is not None:
                    metrics["val_loss"] = epoch_val_loss
                    metrics.update(reid_metrics) # Add re-id metrics to the log
                
                # Use a step that is guaranteed to be after all batch steps for this epoch
                epoch_end_step = (epoch + 1) * len(self.dataloader)
                wandb_logger.log_metrics(metrics, step=epoch_end_step)

            # Early stopping based on patience (now using the monitoring_loss)
            if early_stopping_patience is not None:
                if monitoring_loss < best_monitoring_loss:
                    best_monitoring_loss = monitoring_loss
                    patience_counter = 0
                    logger.debug(f"New best loss: {best_monitoring_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.debug(f"Patience counter: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered due to patience ({patience_counter} epochs without improvement)")
                    break

            # Step the learning rate scheduler (now using the monitoring_loss)
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                self.lr_scheduler.step(monitoring_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate after scheduler step: {current_lr:.6f}")

        logger.info("Training completed successfully")
        return self.model



    def _evaluate_reid_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Computes re-identification metrics (Recall@k and mAP) on the provided dataloader.
        
        This function uses the current model to generate embeddings for all images
        in the dataloader. For triplet datasets, uses only the anchor images for evaluation.
        
        Args:
            dataloader: The dataloader for the validation or test set (can be triplet format).
            
        Returns:
            A dictionary containing the calculated metrics.
        """
        self.model.eval()
        all_embeddings = []
        all_pids = []  # Player IDs
        
        # Generate embeddings for all images in the dataset
        with torch.no_grad():
            for batch_data in dataloader:
                # Handle both triplet format (anchor, positive, negative, label) 
                # and simple format (images, labels)
                if len(batch_data) == 4:
                    # Triplet format: use only anchor images
                    anchor, positive, negative, pids = batch_data
                    images = anchor.to(self.device)
                elif len(batch_data) == 2:
                    # Simple format: use images directly
                    images, pids = batch_data
                    images = images.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
                    
                embeddings = self.model.forward(images)
                all_embeddings.append(embeddings.cpu())
                all_pids.extend(pids.tolist())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_pids = np.array(all_pids)
        
        num_queries = len(all_embeddings)
        
        # Calculate cosine similarity matrix
        embeddings_norm = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.T)
        
        # Initialize metrics
        rank1_correct = 0
        rank5_correct = 0
        rank10_correct = 0
        average_precisions = []

        for i in range(num_queries):
            query_pid = all_pids[i]
            
            # Get similarities for the current query
            query_similarities = similarity_matrix[i, :]
            
            # Exclude the query itself from the gallery
            # A simple way is to zero out its similarity
            query_similarities[i] = -1.0
            
            # Sort by similarity in descending order
            sorted_indices = torch.argsort(query_similarities, descending=True)
            
            # Get the pids of the ranked images
            ranked_pids = all_pids[sorted_indices.numpy()]
            
            # Calculate precision and recall for this query
            num_correct = 0
            precision_at_k = 0.0
            
            # Count true matches at different ranks
            if ranked_pids[0] == query_pid:
                rank1_correct += 1
            if query_pid in ranked_pids[:5]:
                rank5_correct += 1
            if query_pid in ranked_pids[:10]:
                rank10_correct += 1
                
            # Calculate Average Precision for this query
            # This is a bit simplified, but captures the core idea
            for k, pid_k in enumerate(ranked_pids):
                if pid_k == query_pid:
                    num_correct += 1
                    precision_at_k += num_correct / (k + 1.0)
            
            # Avoid division by zero if there are no correct matches
            if num_correct > 0:
                average_precisions.append(precision_at_k / num_correct)
            else:
                average_precisions.append(0.0)

        # Compute final metrics
        recall_1 = rank1_correct / num_queries
        recall_5 = rank5_correct / num_queries
        recall_10 = rank10_correct / num_queries
        mean_average_precision = float(np.mean(average_precisions))
        
        return {
            "recall@1": recall_1,
            "recall@5": recall_5,
            "recall@10": recall_10,
            "mAP": mean_average_precision
        }


    def train_and_save(self, model_class, dataset: Dataset, model_name: str, val_dataset: Optional[Dataset] = None, model_kwargs: Dict[str, Any] = {}) -> Any:
        """
        Complete training pipeline: setup data, setup model, train, and save.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
            dataset: The dataset instance to use (e.g., LacrossePlayerDataset)
            model_name: Name of the model to save in wandb registry
            force_pretrained: If True, ignore saved weights and start fresh
            val_dataset: Optional validation dataset for early stopping and metrics
            
        Returns:
            The trained model
            
        Raises:
            Exception: If any step in the training pipeline fails
        """
        try:
            logger.info("Starting complete training pipeline")
            
            # Setup data
            logger.info("Setting up training data...")
            self.setup_dataloader(dataset)

            if val_dataset is not None:
                logger.info("Setting up validation data...")
                self.setup_dataloader(val_dataset, type='val')

            
            # Setup model
            logger.info("Setting up model...")
            self.setup_model(model_class, model_name=model_name, **model_kwargs)

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
        }
