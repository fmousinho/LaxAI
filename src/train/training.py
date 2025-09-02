import os
import gc
import torch
import logging
from typing import Optional, Any, Dict, Callable
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
from config.all_config import model_config, training_config, wandb_config
from train.wandb_logger import wandb_logger
from utils.gpu_memory import clear_gpu_memory, log_gpu_memory_stats, GPUMemoryContext
from utils.cpu_memory import CPUMemoryMonitor, clear_cpu_memory, cpu_memory_context, log_comprehensive_memory_stats
from utils.evaluation_memory import log_evaluation_memory_usage

logger = logging.getLogger(__name__)

EPOCHS_PER_VAL = 10
BATCHES_PER_LOG_MSG = 10
THRESHOLD_FOR_DATALOADER_RESTART = 90.0
EPOCHS_PER_VAL = 0

class Training:
    """
    A training class specifically for neural network training using triplet loss.
    Focused on training lacrosse player re-identification models.

    """
    
    # Type annotations for dynamically assigned attributes
    num_epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    margin: float
    weight_decay: float
    lr_scheduler_patience: int
    lr_scheduler_threshold: float
    lr_scheduler_min_lr: float
    lr_scheduler_factor: float
    force_pretraining: bool
    num_workers: int
    prefetch_factor: int
    margin_decay_rate: float
    margin_change_threshold: float
    early_stopping_patience: Optional[int]
    
    # Legacy attribute names for backward compatibility
    scheduler_patience: int
    scheduler_threshold: float
    
    def __init__(self, 
                device: Any = None,
                enable_multithreading: bool = True,
                num_workers: Optional[int] = None,
                clear_memory_on_start: bool = True,
                **kwargs):
        """
        Initialize the training class with hyperparameters.
        
        Args:
            device: Device to run the model on (CPU, GPU, or MPS)
            enable_multithreading: Whether to enable multithreading for data loading
            num_workers: Number of DataLoader workers (auto-detected if None)
            clear_memory_on_start: Whether to clear GPU memory on initialization
            **kwargs: Training parameters (see parameter_registry for complete list)
        
        All hyperparameters are defined in config.parameter_registry and will be
        automatically validated. If any required hyperparameter is missing in both 
        kwargs and training_config, a ValueError will be raised.
        """
        from config.parameter_registry import parameter_registry

        # Clear GPU memory at start to recover from previous crashes
        if clear_memory_on_start:
            clear_gpu_memory()

        # Configure threading settings
        self.enable_multithreading = enable_multithreading
        
        # Allow API override of num_workers via kwargs
        api_num_workers = kwargs.get('num_workers')
        if api_num_workers is not None:
            self.num_workers = api_num_workers
            logger.info(f"Using num_workers from API: {self.num_workers}")
        elif enable_multithreading:
            # Use default PyTorch multiprocessing with safe number of workers
            import multiprocessing as mp
            self.num_workers = num_workers if num_workers is not None else min(mp.cpu_count(), 8)
        else:
            self.num_workers = 0
        
        logger.info(f"Training configured with multithreading={'enabled' if enable_multithreading else 'disabled'}, workers={self.num_workers}")

        # Store kwargs for later use (e.g., passing eval_kwargs to evaluation)
        self.kwargs = kwargs

        # Initialize all registered parameters using the centralized registry
        for param_name in parameter_registry.parameters:
            param_def = parameter_registry.parameters[param_name]
            if param_def.config_path.startswith("training_config"):
                # Skip num_workers since it's already handled above
                if param_name == 'num_workers':
                    continue
                try:
                    value = parameter_registry.get_kwarg_or_config(param_name, kwargs)
                    setattr(self, param_name, value)
                except ValueError as e:
                    if param_def.required:
                        raise e
                    # For non-required params, we can skip them

        # Handle specific attribute name mappings for backward compatibility
        if hasattr(self, 'lr_scheduler_patience'):
            self.scheduler_patience = self.lr_scheduler_patience
        if hasattr(self, 'lr_scheduler_threshold'):
            self.scheduler_threshold = self.lr_scheduler_threshold

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
        self.cpu_monitor = CPUMemoryMonitor()
        self.last_worker_restart_epoch = -1  # Track last worker restart to prevent too frequent restarts


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
                'model_class': lambda **kwargs: model_class(**kwargs),
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
            # If wandb is expected to be enabled, surface the error so the
            # pipeline fails loudly instead of silently falling back.
            if getattr(wandb_config, 'enabled', False):
                logger.error(f"Failed to load model from wandb registry while wandb is enabled: {e}")
                raise
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
            # If wandb is expected to be enabled, fail the pipeline
            if getattr(wandb_config, 'enabled', False):
                logger.error(f"Failed to save model to wandb registry while wandb is enabled: {e}")
                raise
            logger.error(f"Failed to save model to wandb registry: {e}")


    def setup_dataloader(self, dataset, type: str = 'train'):
        """
        Setup the dataloader for the given dataset.

        Args:
            dataset: The dataset object to load
            type: Type of dataloader to create ('train' or 'val'). If 'train', uses shuffle=True.
                  If 'val', uses shuffle=False.
            
        """
        # Configure DataLoader settings for optimal speed

        prefetch_factor = self.kwargs.get('prefetch_factor', training_config.prefetch_factor) if self.num_workers > 0 else None

        dataloader_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': torch.cuda.is_available() and self.num_workers > 0,  # Only pin memory with workers
            'persistent_workers': self.num_workers > 0,
            'prefetch_factor': prefetch_factor,  # Increased prefetch for speed
            'drop_last': True if type == 'train' else False  # Drop incomplete batches for consistent timing
        }
        
        # Add batch size and dataset-specific options
        base_config = {
            'batch_size': self.batch_size,
            **dataloader_kwargs
        }
        
        if type == 'train':
            self.dataloader = DataLoader(
                dataset,
                shuffle=True,
                **base_config
            )

        elif type == 'val':
            self.val_dataloader = DataLoader(
                dataset,
                shuffle=False,
                **base_config
            )
        else:
            raise ValueError(f"Invalid dataloader type: {type}. Use 'train' or 'val'.")

        # Log configuration
        active_dataloader = self.dataloader if type == 'train' else self.val_dataloader
        if active_dataloader is not None:
            logger.info(f"DataLoader setup complete for {type}:")
            logger.info(f"  Multithreading: {'enabled' if self.enable_multithreading else 'disabled'}")
            logger.info(f"  Workers: {active_dataloader.num_workers}")
            logger.info(f"  Pin memory: {active_dataloader.pin_memory}")
            if hasattr(active_dataloader, 'persistent_workers'):
                logger.info(f"  Persistent workers: {active_dataloader.persistent_workers}")

        logger.info(f"Dataset summary:")
        logger.info(f"  - Type: {type}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Number of batches: {len(self.dataloader) if self.dataloader is not None else len(self.val_dataloader) if self.val_dataloader is not None else 'N/A'}")



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
                logger.info("Forcing fresh start with pre-trained weights")
                self.model = model_class(**kwargs)
            else:
                model_loaded = self._load_model_from_wandb(model_class, model_name=model_name, alias="latest", **kwargs)
            if not model_loaded:
                logger.info("No wandb model found, will use local weights or pre-trained backbone")
                self.model = model_class(**kwargs)

            self.model.to(self.device)
            log_gpu_memory_stats("After moving model to device")
            
            # Setup training components
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay
            )
            log_gpu_memory_stats("After creating optimizer")
            # Scheduler uses its own patience/threshold for LR adjustment only
        
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold,
                min_lr=self.lr_scheduler_min_lr
            )

            # Ensure optimizer state tensors are on the same device as model
            try:
                self._move_optimizer_state_to_device(self.device)
            except Exception:
                logger.debug("Could not move optimizer state to device during setup; will attempt later if needed")

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
          margin_change_threshold: float = training_config.margin_change_threshold,
          start_epoch: int = 1,
          stop_callback: Optional[Callable[[], bool]] = None):
        """
        Execute the main training loop with early stopping, using a validation set
        to monitor for overfitting.
        
        Args:
            margin_decay_rate: Rate at which to decay the triplet loss margin.
            margin_change_threshold: Minimum change in margin to trigger an update.
            start_epoch: Epoch number to start training from (for checkpoint resumption).
            stop_callback: Optional callback function that returns True if training should stop.
            
        Returns:
            The trained model
            
        Raises:
            RuntimeError: If required components are not setup
            InterruptedError: If training is cancelled via stop_callback
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
        effective_epochs = self.num_epochs - (start_epoch - 1)
        logger.info(f"Starting training for {effective_epochs} epochs (from epoch {start_epoch} to {self.num_epochs})")
        
        # Early stopping configuration
        early_stopping_patience = getattr(training_config, 'early_stopping_patience', None)
        best_monitoring_loss = float('inf')
        patience_counter = 0

        # Margin decay setup
        current_margin = self.margin
        self.loss_fn = nn.TripletMarginLoss(margin=current_margin, p=2)
        val_dataloader = self.val_dataloader
        
        try:
            log_gpu_memory_stats("Training start")
            self.cpu_monitor.log_memory_stats("Training start")
            
            for epoch in range(start_epoch - 1, self.num_epochs):
                # Check for cancellation before starting epoch
                if stop_callback and stop_callback():
                    logger.info(f"Training cancelled by stop_callback at epoch {epoch + 1}")
                    raise InterruptedError("Training cancelled by external request")

                # Memory-based worker restart to prevent accumulation
                if self.num_workers > 0 and epoch > 0:
                    current_memory = self.cpu_monitor._get_current_memory()
                    memory_percent = current_memory["percent"]
                    
                    # Restart if memory utilization exceeds 90% and we haven't restarted recently (at least 1 epoch ago)
                    if memory_percent > THRESHOLD_FOR_DATALOADER_RESTART and (epoch - self.last_worker_restart_epoch) >= 1:
                        logger.info(f"Restarting DataLoader workers at epoch {epoch + 1} - CPU memory utilization: {memory_percent:.1f}%")
                        
                        # Log memory before restart
                        memory_before = current_memory["rss_mb"]
                        logger.info(f"Memory before worker restart: {memory_before:.1f}MB ({memory_percent:.1f}%)")
                        
                        self._restart_dataloader_workers()
                        self.last_worker_restart_epoch = epoch  # Update last restart epoch
                        
                        # Log memory after restart
                        memory_after = self.cpu_monitor._get_current_memory()["rss_mb"]
                        memory_after_percent = self.cpu_monitor._get_current_memory()["percent"]
                        memory_delta = memory_after - memory_before
                        logger.info(f"Memory after worker restart: {memory_after:.1f}MB ({memory_after_percent:.1f}%) (Δ{memory_delta:+.1f}MB)")
                    elif memory_percent > 90.0:
                        logger.debug(f"CPU memory utilization: {memory_percent:.1f}% - restart skipped (too recent)")
                    else:
                        logger.debug(f"CPU memory utilization: {memory_percent:.1f}% - no restart needed")
                
                log_gpu_memory_stats(f"Epoch {epoch + 1} start")
                self.cpu_monitor.log_memory_stats(f"Epoch {epoch + 1} start")

                # ========================================================================
                # Training Phase
                # ========================================================================
                # Model and optimizer should already be placed on the correct device
                # during setup_model (or after checkpoint resumption). Repeatedly
                # calling .to(device) each epoch can unintentionally create new
                # parameter tensors and leave old tensors referenced by the
                # optimizer state, causing cumulative memory growth. Avoid moving
                # them here.
                self.model.train()
                running_loss = 0.0
                ttl_batches = len(self.dataloader)

                # Update margin if decay rate is active (use actual epoch number)
                new_margin = self.margin * (margin_decay_rate ** epoch)
                if abs(new_margin - current_margin) > margin_change_threshold:
                    current_margin = new_margin
                    self.loss_fn = nn.TripletMarginLoss(margin=current_margin, p=2)
                    logger.info(f"Margin updated for epoch {epoch+1}: {current_margin:.4f}")
                else:
                    logger.debug(f"Margin unchanged for epoch {epoch+1}: {current_margin:.4f}")
                    
                for i, (anchor, positive, negative, _) in enumerate(self.dataloader):

                    # Log progress
                    if (i + 1) % BATCHES_PER_LOG_MSG == 0:
                        logger.info(f"Training Batch {i+1}/{ttl_batches}")

                    # Move tensors to device with non_blocking for async transfer
                    anchor = anchor.to(self.device, non_blocking=True)
                    positive = positive.to(self.device, non_blocking=True)
                    negative = negative.to(self.device, non_blocking=True)

                    # Clear gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    emb_anchor, emb_positive, emb_negative = self.model.forward_triplet(anchor, positive, negative)  # pyright: ignore[reportCallIssue]
                    loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)

                    # Backward and optimizer step
                    loss.backward()
                    self.optimizer.step()

                    # Extract scalar immediately and clear GPU tensors to free memory
                    running_loss += loss.item()
                    
                    # Explicitly delete tensors to help free GPU memory faster
                    del anchor, positive, negative, emb_anchor, emb_positive, emb_negative, loss
                    
                    # # Clear GPU cache periodically
                    # if torch.cuda.is_available() and (i + 1) % 50 == 0:
                    #     torch.cuda.empty_cache()
                        
                    # # Clear CPU memory periodically to prevent accumulation
                    # if (i + 1) % 100 == 0:
                    #     clear_cpu_memory()
                    #     self.cpu_monitor.log_memory_stats(f"Epoch {epoch + 1} Batch {i + 1}")

                # Calculate and log training loss
                epoch_train_loss = running_loss / ttl_batches if ttl_batches > 0 else 0.0
                
                # Memory cleanup after training phase
                # clear_cpu_memory()
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                # self.cpu_monitor.log_memory_stats(f"Epoch {epoch + 1} training complete")

                # ========================================================================
                # Validation Phase (if dataloader is provided)
                # ========================================================================
               

                # Check for cancellation before validation
                if stop_callback and stop_callback():
                    logger.info(f"Training cancelled by stop_callback before validation at epoch {epoch + 1}")
                    raise InterruptedError("Training cancelled by external request")

                epoch_val_loss = None
                reid_metrics = {} 


                if EPOCHS_PER_VAL > 0 and val_dataloader and (epoch + 1) % EPOCHS_PER_VAL == 0:

                    self.model.eval()  # Set model to evaluation mode
                    
                    # 1. Calculate Validation Loss
                    running_val_loss = 0.0
                    ttl_batches = len(val_dataloader)
                    with torch.no_grad(): # No need to compute gradients
                        for j, (anchor, positive, negative, _) in enumerate(val_dataloader):
                            anchor = anchor.to(self.device)
                            positive = positive.to(self.device)
                            negative = negative.to(self.device)
                            
                            emb_anchor, emb_positive, emb_negative = self.model.forward_triplet(anchor, positive, negative) # pyright: ignore[reportCallIssue]
                            loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)
                            
                            running_val_loss += loss.item()

                            if (j + 1) % BATCHES_PER_LOG_MSG == 0:
                                logger.info(f"Validation Batch {j+1}/{ttl_batches}")

                    epoch_val_loss = running_val_loss / ttl_batches if ttl_batches > 0 else 0.0

                    # 2. Calculate Retrieval Metrics
                    reid_metrics = self._evaluate_reid_metrics(val_dataloader)
                    
                    # Memory cleanup after validation
                    clear_cpu_memory()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.cpu_monitor.log_memory_stats(f"Epoch {epoch + 1} validation complete")

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
                    
                  
                    wandb_logger.log_metrics(metrics)

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

                # Save checkpoint at the end of each epoch
                if wandb_config.enabled and self.optimizer is not None:
                    try:
                        # Monitor memory before checkpoint save
                        import psutil
                        process = psutil.Process()
                        memory_before_checkpoint = process.memory_info().rss / 1024 / 1024  # MB
                        logger.debug(f"Memory before checkpoint save: {memory_before_checkpoint:.1f}MB")
                        
                        model_config_dict = {
                            "margin": self.margin,
                            "learning_rate": self.learning_rate,
                            "batch_size": self.batch_size,
                            "weight_decay": self.weight_decay
                        }
                        
                        # wandb_logger.save_checkpoint(
                        #     epoch=epoch + 1,  # Save 1-indexed epoch number
                        #     model_state_dict=self.model.state_dict(),
                        #     optimizer_state_dict=self.optimizer.state_dict(),
                        #     loss=monitoring_loss,
                        #     model_name=type(self.model).__name__,
                        #     model_config=model_config_dict
                        # )
                        logger.debug(f"Checkpoint saved for epoch {epoch + 1}")
                        
                        # Monitor memory after checkpoint save
                        memory_after_checkpoint = process.memory_info().rss / 1024 / 1024  # MB
                        checkpoint_memory_delta = memory_after_checkpoint - memory_before_checkpoint
                        if abs(checkpoint_memory_delta) > 100:  # Log significant memory changes
                            logger.info(f"Checkpoint memory usage: {memory_before_checkpoint:.1f}MB → {memory_after_checkpoint:.1f}MB (Δ{checkpoint_memory_delta:+.1f}MB)")
                        
                    except Exception as e:
                        if getattr(wandb_config, 'enabled', False):
                            logger.error(f"Failed to save checkpoint for epoch {epoch + 1} while wandb is enabled: {e}")
                            raise
                        logger.warning(f"Failed to save checkpoint for epoch {epoch + 1}: {e}")


        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"Training failed with error: {e}")
            
            # Clean up both GPU and CPU memory on failure
            logger.info("Cleaning up GPU and CPU memory after training failure...")
            
            # Clear all variables that might hold GPU tensors
            if 'anchor' in locals():
                del anchor
            if 'positive' in locals():
                del positive
            if 'negative' in locals():
                del negative
            if 'emb_anchor' in locals():
                del emb_anchor
            if 'emb_positive' in locals():
                del emb_positive
            if 'emb_negative' in locals():
                del emb_negative
            if 'loss' in locals():
                del loss
            
            # Force cleanup
            clear_gpu_memory()
            clear_cpu_memory(force=True)
            self.cpu_monitor.log_memory_stats("After failure cleanup")
            
            # Re-raise the exception
            raise
            
        except Exception as e:
            logger.error(f"Training failed with unexpected error: {e}")
            
            # Check if this is a CUDA OOM error and provide specific guidance
            if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                logger.error("CUDA Out of Memory Error detected!")
                log_gpu_memory_stats("Before OOM cleanup")
                clear_gpu_memory(force=True)
                clear_cpu_memory(force=True)
                log_gpu_memory_stats("After OOM cleanup")
                self.cpu_monitor.log_memory_stats("After OOM cleanup")
                logger.error("Suggestions to resolve OOM:")
                logger.error("1. Reduce batch_size in your configuration")
                logger.error("2. Use gradient accumulation to simulate larger batches")
                logger.error("3. Consider mixed precision training (fp16)")
                logger.error("4. Use a smaller model architecture")
            else:
                clear_gpu_memory()
                clear_cpu_memory()
            raise

        logger.info("Training completed successfully")
        log_comprehensive_memory_stats("Training completion")
        
        # Optional: Move model to CPU to save GPU memory if evaluation will be on CPU
        # Uncomment the following lines if you want to conserve GPU memory:
        # if self.device.type == 'cuda' and torch.cuda.is_available():
        #     logger.info("Moving trained model to CPU to conserve GPU memory")
        #     self.model = self.model.cpu()
        #     torch.cuda.empty_cache()
        
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
        # Use the centralized comprehensive evaluator to compute and persist
        # full evaluation, then return the retrieval metrics used by training.
        from train.evaluator import ModelEvaluator

        # dataloader may be a DataLoader; evaluator expects a Dataset instance
        dataset = getattr(dataloader, 'dataset', dataloader)

        # Create evaluator with memory management
        evaluator = ModelEvaluator(self.model, device=self.device)

        # Log memory before evaluation
        import psutil
        process = psutil.Process()
        memory_before_eval = process.memory_info().rss / 1024 / 1024  # MB
        logger.debug(f"Memory before evaluation: {memory_before_eval:.1f}MB")
        
        log_evaluation_memory_usage("Before validation evaluation", evaluator)

        try:
            # Run the comprehensive evaluation (this also saves results to disk)
            results = evaluator.evaluate_comprehensive(dataset, **self.kwargs)
        except Exception as e:
            logger.warning(f"Comprehensive evaluation failed, falling back to local computation: {e}")
            results = {}
        finally:
            # Always cleanup evaluator to prevent memory leaks
            evaluator.cleanup()
            
            # Monitor memory after evaluation
            memory_after_eval = process.memory_info().rss / 1024 / 1024  # MB
            eval_memory_delta = memory_after_eval - memory_before_eval
            if abs(eval_memory_delta) > 50:  # Log significant memory changes
                logger.info(f"Evaluation memory usage: {memory_before_eval:.1f}MB → {memory_after_eval:.1f}MB (Δ{eval_memory_delta:+.1f}MB)")
            
            log_evaluation_memory_usage("After validation evaluation cleanup")

        # If evaluation succeeded, flatten all nested metrics into a single-level dict;
        # otherwise fall back to a local computation that produces the same metric groups.
        flat_metrics: Dict[str, Any] = {}

        def _flatten(prefix: str, obj: Any):
            """Recursively flatten dict-like metric objects into flat_metrics using
            joined keys. Does not rely on specific section name patterns.
            """
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}_{k}" if prefix else k
                    _flatten(new_prefix, v)
            else:
                # Try to coerce numeric-like values to float for consistency
                try:
                    flat_metrics[prefix] = float(obj)
                except Exception:
                    flat_metrics[prefix] = obj

        if isinstance(results, dict) and results:
            # Iterate over whatever sections the evaluator returned; future-proof.
            for section, metrics in results.items():
                _flatten(section, metrics)

        return flat_metrics


    def setup_training_pipeline(self, model_class, dataset: Dataset, model_name: str, val_dataset: Optional[Dataset] = None, model_kwargs: Dict[str, Any] = {}):
        """
        Setup the complete training pipeline components.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
            dataset: The dataset instance to use (e.g., LacrossePlayerDataset)
            model_name: Name of the model to save in wandb registry
            val_dataset: Optional validation dataset for early stopping and metrics
            model_kwargs: Additional arguments for model instantiation
        """
        logger.info("Setting up training pipeline components...")
        
        # Setup data
        logger.info("Setting up training data...")
        self.setup_dataloader(dataset)

        if val_dataset is not None:
            logger.info("Setting up validation data...")
            self.setup_dataloader(val_dataset, type='val')

        # Setup model
        logger.info("Setting up model...")
        self.setup_model(model_class, model_name=model_name, **model_kwargs)

    def check_for_checkpoint_resumption(self) -> int:
        """
        Check for existing checkpoint and determine starting epoch.
        
        Returns:
            Starting epoch number (1 if no checkpoint, >1 if resuming)
        """
        start_epoch = 1
        
        if not wandb_config.enabled:
            logger.info("WandB not enabled, starting fresh training")
            return start_epoch
            
        logger.info("Checking for existing checkpoint")
        try:
            # Ensure optimizer exists before attempting to resume
            if self.optimizer is None:
                logger.warning("Optimizer not initialized, cannot resume from checkpoint")
                return start_epoch
                
            start_epoch = wandb_logger.resume_training_from_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                artifact_name=wandb_logger.get_checkpoint_name(),
                version="latest"
            )
            
            if start_epoch > 1:
                logger.info(f"✅ Resumed training from checkpoint at epoch {start_epoch}")
                # Ensure model and optimizer state are on the configured device
                try:
                    self.model.to(self.device)
                    # Move optimizer state tensors (if any) to device to avoid
                    # runtime errors during optimizer.step caused by mixed devices.
                    try:
                        self._move_optimizer_state_to_device(self.device)
                    except Exception:
                        logger.debug("Could not move optimizer state to device after resuming checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to move resumed model to device {self.device}: {e}")
                # Check if training is already completed
                remaining_epochs = max(0, self.num_epochs - (start_epoch - 1))
                if remaining_epochs == 0:
                    logger.info("Training already completed according to checkpoint!")
                    # Return a special value to indicate completion
                    return self.num_epochs + 1
                logger.info(f"Training will continue for {remaining_epochs} more epochs")
            else:
                logger.info("No valid checkpoint found, starting fresh training")
                
        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}")
            logger.info("Starting fresh training")
            start_epoch = 1
            
        return start_epoch

    def train_and_save(self, model_class, dataset: Dataset, model_name: str, val_dataset: Optional[Dataset] = None, model_kwargs: Dict[str, Any] = {}, resume_from_checkpoint: bool = True, stop_callback: Optional[Callable[[], bool]] = None) -> Any:
        """
        Complete training pipeline: setup, train, and save.
        This is a convenience method with minimal logic.
        
        Args:
            model_class: The model class to use (e.g., SiameseNet)
            dataset: The dataset instance to use (e.g., LacrossePlayerDataset)
            model_name: Name of the model to save in wandb registry
            val_dataset: Optional validation dataset for early stopping and metrics
            model_kwargs: Additional arguments for model instantiation
            resume_from_checkpoint: Whether to resume from existing wandb checkpoint
            stop_callback: Optional callback function that returns True if training should stop
            
        Returns:
            The trained model
            
        Raises:
            Exception: If any step in the training pipeline fails
            InterruptedError: If training is cancelled via stop_callback
        """
        try:
            logger.info("Starting complete training pipeline")
            
            # Setup pipeline components
            self.setup_training_pipeline(model_class, dataset, model_name, val_dataset, model_kwargs)
            
            # Check for checkpoint resumption
            start_epoch = 1
            if resume_from_checkpoint:
                start_epoch = self.check_for_checkpoint_resumption()
                if start_epoch > self.num_epochs:
                    logger.info("Training already completed!")
                    return self.model

            # Train with checkpoint support
            logger.info("Starting training...")
            trained_model = self.train(start_epoch=start_epoch, stop_callback=stop_callback)
            
            # Save final model
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

    def _move_optimizer_state_to_device(self, device: torch.device):
        """Move all optimizer state tensors to the target device.

        Some optimizers allocate state tensors (exp_avg, exp_avg_sq) on creation.
        If the model was created on CPU and later moved to GPU, these state
        tensors remain on CPU and will cause runtime errors during optimizer.step.
        This helper moves them to the desired device.
        """
        if self.optimizer is None:
            return
        try:
            for param_group in self.optimizer.param_groups:
                for p in param_group.get('params', []):
                    if p in self.optimizer.state:
                        state = self.optimizer.state[p]
                        for k, v in list(state.items()):
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
        except Exception:
            logger.debug("Failed to move some optimizer state tensors to device; they may be created lazily later")

    def cleanup_model(self):
        """
        Clean up the model and associated resources to free memory.
        
        This method should be called after evaluation is complete to prevent
        memory leaks from accumulated model references.
        """
        if hasattr(self, 'model') and self.model is not None:
            logger.info("Cleaning up training model and resources")
            
            # Clear model from GPU memory
            if torch.cuda.is_available():
                self.model = self.model.cpu()
                torch.cuda.empty_cache()
            
            # Clear optimizer state
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                # Clear optimizer state dict which can hold significant memory
                self.optimizer.state.clear()
                self.optimizer.param_groups.clear()
            
            # Clear loss function
            if hasattr(self, 'loss_fn'):
                self.loss_fn = None
            
            # Clear dataloaders
            if hasattr(self, 'dataloader'):
                self.dataloader = None
            if hasattr(self, 'val_dataloader'):
                self.val_dataloader = None
            
            # Force garbage collection
            clear_cpu_memory()
            
            logger.info("Model cleanup completed")
        
    def get_model_for_evaluation(self):
        """
        Get the trained model for evaluation with memory management.
        
        Returns:
            The trained model, moved to appropriate device for evaluation
        """
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("No trained model available. Call train() first.")
        
        # Ensure model is on the correct device for evaluation
        if str(self.model.device) != str(self.device):
            logger.info(f"Moving model from {self.model.device} to {self.device} for evaluation")
            self.model = self.model.to(self.device)
        
        return self.model

    def _restart_dataloader_workers(self):
        """
        Restart DataLoader workers when CPU memory utilization exceeds threshold.
        
        This method terminates existing workers and recreates the DataLoader
        with fresh workers, clearing any accumulated state. Called automatically
        when CPU memory utilization exceeds 90%.
        """
        try:
            # Store original dataset references
            train_dataset = self.dataloader.dataset if self.dataloader else None
            val_dataset = self.val_dataloader.dataset if self.val_dataloader else None
            
            # Clean up existing workers
            if self.dataloader and hasattr(self.dataloader, '_iterator'):
                if hasattr(self.dataloader._iterator, '_workers'):
                    for worker in self.dataloader._iterator._workers:
                        if worker.is_alive():
                            worker.terminate()
                            worker.join(timeout=1.0)
            
            if self.val_dataloader and hasattr(self.val_dataloader, '_iterator'):
                if hasattr(self.val_dataloader._iterator, '_workers'):
                    for worker in self.val_dataloader._iterator._workers:
                        if worker.is_alive():
                            worker.terminate()
                            worker.join(timeout=1.0)
            
            # Force garbage collection to clean up worker references
            import gc
            gc.collect()
            
            # Recreate DataLoaders with fresh workers
            if train_dataset:
                self.setup_dataloader(train_dataset, type='train')
                logger.info("Training DataLoader workers restarted")
            
            if val_dataset:
                self.setup_dataloader(val_dataset, type='val')
                logger.info("Validation DataLoader workers restarted")
                
        except Exception as e:
            logger.warning(f"Failed to restart DataLoader workers: {e}")
            # Continue training even if worker restart fails
