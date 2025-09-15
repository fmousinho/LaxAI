import gc
import logging
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from evaluator import (ModelEvaluator, calculate_embedding_variance,
                       calculate_gradient_norm,
                       calculate_intra_inter_distances,
                       calculate_triplet_mining_efficiency)
from torch.utils.data import DataLoader, Dataset
from wandb_logger import wandb_logger

from shared_libs.config.all_config import (model_config, training_config,
                                           wandb_config)
from shared_libs.utils.cpu_memory import (CPUMemoryMonitor, clear_cpu_memory,
                                          cpu_memory_context,
                                          log_comprehensive_memory_stats)
from shared_libs.utils.dataloader_memory import worker_init_fn
from shared_libs.utils.gpu_memory import (GPUMemoryContext, clear_gpu_memory,
                                          log_gpu_memory_stats)

logger = logging.getLogger(__name__)

# Constants
BATCHES_PER_LOG_MSG = 10
EPOCHS_PER_VAL = 0
EPOCHS_PER_DATALOADER_RESTART = 10
THRESHOLD_FOR_DATALOADER_RESTART = 90.0  # Memory usage percentage threshold


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
        from parameter_registry import parameter_registry

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

        # Log initial device and memory state
        logger.info(f"Training device: {self.device}")
        if torch.cuda.is_available() and self.device.type == 'cuda':
            log_gpu_memory_stats("Device initialization")

        self.model: nn.Module
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.dataloader = None
        self.val_dataloader = None
        self.cpu_monitor = CPUMemoryMonitor()
        
        # Memory leak tracking
        self.epoch_memory_baseline = None
        self.memory_leak_threshold_mb = 100  # Alert if memory increases by more than 100MB per epoch
        
        # GPU cache management strategy
        # For GPU-enabled Cloud Run, use conservative cache clearing to preserve performance
        self.gpu_cache_threshold = kwargs.get('gpu_cache_threshold', 0.85)  # Clear cache when >85% GPU memory used
        self.conservative_gpu_cache = kwargs.get('conservative_gpu_cache', True)  # Default to conservative for Cloud Run GPU


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
                
                # Log memory after model loading from WandB
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    log_gpu_memory_stats("After WandB model load")
                
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
        
        # Skip saving if WandB is not enabled
        if not wandb_logger.enabled:
            logger.info("WandB not enabled, skipping model save to registry")
            return
        
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

            if wandb_logger.enabled:
                wandb_logger.save_model_to_registry(
                    model=self.model,
                    collection_name=model_name,
                    metadata=metadata
                )
                logger.info(f"✓ Model saved to wandb registry: {model_name}:latest")
            else:
                logger.info(f"WandB not enabled, skipping model save to registry")
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
        
        # Add worker initialization function to suppress logs and show custom messages
        if self.num_workers > 0:
            dataloader_kwargs['worker_init_fn'] = worker_init_fn
        
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

        # Validate that we have at least 1 batch
        active_dataloader = self.dataloader if type == 'train' else self.val_dataloader
        if active_dataloader is not None and len(active_dataloader) == 0:
            # Safely get dataset size with proper type checking
            dataset_size = 'unknown'
            if hasattr(active_dataloader.dataset, '__len__'):
                try:
                    dataset_size = len(active_dataloader.dataset)  # type: ignore[arg-type]
                except (TypeError, AttributeError):
                    dataset_size = 'unknown'
            raise ValueError(
                f"Insufficient data for training! "
                f"Dataset has {dataset_size} samples, but batch size is {self.batch_size}. "
                f"This results in 0 batches for {type} (drop_last={'True' if type == 'train' else 'False'}). "
                f"Please either: "
                f"1) Reduce batch_size to be smaller than dataset size, or "
                f"2) Add more data to the dataset, or "
                f"3) For training, consider using a smaller batch_size"
            )

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

            # Log memory before model operations
            if torch.cuda.is_available() and self.device.type == 'cuda':
                log_gpu_memory_stats("Before model setup")

            # Try to load from wandb registry first (unless forcing pretrained)
            if self.force_pretraining:
                logger.info("Forcing fresh start with pre-trained weights")
                self.model = model_class(**kwargs)
            else:
                model_loaded = self._load_model_from_wandb(model_class, model_name=model_name, alias="latest", **kwargs)
            if not model_loaded:
                logger.info("No wandb model found, will use local weights or pre-trained backbone")
                self.model = model_class(**kwargs)

            # Avoid unnecessary device movement to prevent temporary memory doubling
            if str(self.model.device) != str(self.device):
                logger.info(f"Moving model from {self.model.device} to {self.device}")
                self.model.to(self.device)
            else:
                logger.debug(f"Model already on correct device: {self.device}")
            log_gpu_memory_stats("After moving model to device")
            
            # Enable backbone fine-tuning for DINOv3 models
            if hasattr(self.model, 'enable_backbone_fine_tuning') and callable(getattr(self.model, 'enable_backbone_fine_tuning', None)):
                try:
                    self.model.enable_backbone_fine_tuning(unfreeze_layers=2)  # pyright: ignore[reportCallIssue]
                    logger.info("Enabled backbone fine-tuning for DINOv3 model")
                except Exception as e:
                    logger.warning(f"Could not enable backbone fine-tuning: {e}")
            else:
                logger.debug("enable_backbone_fine_tuning is not callable or not present on model")
            
            # Setup training components with improved loss function
            # Use TripletMarginLoss with distance weighting for better convergence
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')
            
            # Use AdamW optimizer which is better for fine-tuning transformers
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),  # Default AdamW betas
                eps=1e-8  # Slightly higher epsilon for stability
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
        
        # Log memory state at training start
        if torch.cuda.is_available() and self.device.type == 'cuda':
            log_gpu_memory_stats("Training start")
        
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

                # Periodic aggressive memory cleanup to prevent accumulation
                if (epoch + 1) % 2 == 0:  # Every 2 epochs
                    try:
                        self._aggressive_memory_cleanup(f"End of epoch {epoch + 1}")
                    except Exception as cleanup_error:
                        logger.debug(f"Aggressive memory cleanup warning: {cleanup_error}")
                
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

                # Initialize metric accumulators for training progress tracking
                running_embedding_variance = 0.0
                running_intra_distance = 0.0
                running_inter_distance = 0.0
                running_margin_satisfaction = 0.0
                running_hard_triplets = 0.0
                running_easy_triplets = 0.0
                running_mining_efficiency = 0.0
                running_grad_norm = 0.0

                # Update margin if decay rate is active (use actual epoch number)
                new_margin = self.margin * (margin_decay_rate ** epoch)
                if abs(new_margin - current_margin) > margin_change_threshold:
                    current_margin = new_margin
                    self.loss_fn = nn.TripletMarginLoss(margin=current_margin, p=2)
                    logger.info(f"Margin updated for epoch {epoch+1}: {current_margin:.4f}")
                else:
                    logger.debug(f"Margin unchanged for epoch {epoch+1}: {current_margin:.4f}")
                    
                # Add warmup phase for first few epochs
                if epoch < 5:  # Warmup for first 5 epochs
                    warmup_factor = min(1.0, (epoch + 1) / 5.0)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate * warmup_factor
                    logger.debug(f"Warmup LR for epoch {epoch+1}: {self.learning_rate * warmup_factor:.2e}")
                    
                for i, (anchor, positive, negative, _) in enumerate(self.dataloader):

                    # Log progress
                    if (i + 1) % BATCHES_PER_LOG_MSG == 0:
                        logger.info(f"Training Batch {i+1}/{ttl_batches}")

                    # ------------------------------------------------------------------
                    # Mid-epoch cancellation check
                    # The web/API cancellation sets a stop flag on the Pipeline which
                    # propagates here via stop_callback. Previously we only checked at
                    # epoch boundaries, causing long waits for large epochs. We now
                    # check every batch (fast) but still keep operations lightweight.
                    # ------------------------------------------------------------------
                    if stop_callback and stop_callback():
                        logger.info(
                            f"Training cancelled by stop_callback mid-epoch at epoch {epoch + 1}, batch {i + 1}"
                        )
                        # Raise InterruptedError to trigger upstream cleanup logic
                        raise InterruptedError("Training cancelled by external request (mid-epoch)")
                    

                    # Move tensors to device with non_blocking for async transfer
                    anchor = anchor.to(self.device, non_blocking=True)
                    positive = positive.to(self.device, non_blocking=True)
                    negative = negative.to(self.device, non_blocking=True)

                    # Clear gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    emb_anchor, emb_positive, emb_negative = self.model.forward_triplet(anchor, positive, negative)  # pyright: ignore[reportCallIssue]
                    loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)

                    # Calculate training progress metrics
                    # These metrics help diagnose why loss might be staying near margin
                    concatenated_embeddings = torch.cat([emb_anchor, emb_positive, emb_negative], dim=0)
                    embedding_variance = calculate_embedding_variance(concatenated_embeddings)
                    distance_metrics = calculate_intra_inter_distances(emb_anchor, emb_positive, emb_negative)
                    mining_metrics = calculate_triplet_mining_efficiency(emb_anchor, emb_positive, emb_negative, current_margin)

                    # Backward and optimizer step
                    loss.backward()
                    grad_norm = calculate_gradient_norm(self.model)
                    self.optimizer.step()

                    # Extract scalar immediately and clear GPU tensors to free memory
                    loss_value = loss.item()
                    running_loss += loss_value

                    # Accumulate training progress metrics
                    # 1. Embedding variance - measures feature diversity (high >0.1 is good)
                    running_embedding_variance += embedding_variance
                    # 2. Intra/inter distances - intra should be small (<0.5), inter large (>1.0)
                    running_intra_distance += distance_metrics['intra_class_distance']
                    running_inter_distance += distance_metrics['inter_class_distance']
                    # 3. Margin satisfaction - ratio of triplets satisfying margin constraint (>0.8 is good)
                    running_margin_satisfaction += distance_metrics['triplet_margin_satisfaction']
                    # 4. Triplet mining efficiency - higher ratio (>0.7) indicates better triplet selection
                    running_hard_triplets += mining_metrics['hard_triplets_ratio']
                    running_easy_triplets += mining_metrics['easy_triplets_ratio']
                    running_mining_efficiency += mining_metrics['mining_efficiency']
                    # 5. Gradient norm - monitors for explosion (>1000) or vanishing (<0.01)
                    running_grad_norm += grad_norm
                    
                    # Explicitly delete ALL tensors and intermediate results to prevent accumulation
                    del anchor, positive, negative, emb_anchor, emb_positive, emb_negative, loss
                    del concatenated_embeddings  # Clean up the concatenated tensor
                    
                    # Establish memory baseline after first few batches of first epoch
                    # This accounts for model loading, optimizer state creation, and initial allocations
                    if epoch == 0 and i == 10 and self.epoch_memory_baseline is None:  # After 5th batch of first epoch
                        import psutil
                        process = psutil.Process()
                        self.epoch_memory_baseline = process.memory_info().rss / 1024 / 1024  # MB
                        logger.info(f"Memory baseline established after first 5 batches: {self.epoch_memory_baseline:.1f}MB")
                    

                # Calculate and log training loss
                epoch_train_loss = running_loss / ttl_batches if ttl_batches > 0 else 0.0

                # Calculate average training progress metrics
                epoch_embedding_variance = running_embedding_variance / ttl_batches if ttl_batches > 0 else 0.0
                epoch_intra_distance = running_intra_distance / ttl_batches if ttl_batches > 0 else 0.0
                epoch_inter_distance = running_inter_distance / ttl_batches if ttl_batches > 0 else 0.0
                epoch_margin_satisfaction = running_margin_satisfaction / ttl_batches if ttl_batches > 0 else 0.0
                epoch_hard_triplets = running_hard_triplets / ttl_batches if ttl_batches > 0 else 0.0
                epoch_easy_triplets = running_easy_triplets / ttl_batches if ttl_batches > 0 else 0.0
                epoch_mining_efficiency = running_mining_efficiency / ttl_batches if ttl_batches > 0 else 0.0
                epoch_grad_norm = running_grad_norm / ttl_batches if ttl_batches > 0 else 0.0
                

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
                    
                    # 1. Calculate Validation Loss with aggressive memory management
                    running_val_loss = 0.0
                    ttl_batches = len(val_dataloader)
                    with torch.no_grad(): # No need to compute gradients
                        for j, (anchor, positive, negative, _) in enumerate(val_dataloader):
                            anchor = anchor.to(self.device, non_blocking=True)
                            positive = positive.to(self.device, non_blocking=True)
                            negative = negative.to(self.device, non_blocking=True)
                            
                            emb_anchor, emb_positive, emb_negative = self.model.forward_triplet(anchor, positive, negative) # pyright: ignore[reportCallIssue]
                            loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)
                            
                            # Extract scalar value immediately
                            loss_value = loss.item()
                            running_val_loss += loss_value

                            # Explicit cleanup of validation tensors
                            del anchor, positive, negative, emb_anchor, emb_positive, emb_negative, loss

                            if (j + 1) % BATCHES_PER_LOG_MSG == 0:
                                logger.info(f"Validation Batch {j+1}/{ttl_batches}")
                                
                            # Periodic memory cleanup during validation
                            if (j + 1) % 50 == 0:
                                if torch.cuda.is_available():
                                    # Use smart GPU cache clearing with conservative approach
                                    self._smart_gpu_cache_clear(force=False, context=f"validation_batch_{j+1}")

                    epoch_val_loss = running_val_loss / ttl_batches if ttl_batches > 0 else 0.0

                    # 2. Calculate Retrieval Metrics with memory monitoring
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        log_gpu_memory_stats("Before evaluation")
                    self.cpu_monitor.log_memory_stats("Before evaluation")
                    
                    reid_metrics = self._evaluate_reid_metrics(val_dataloader)
                    
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        log_gpu_memory_stats("After evaluation")
                    self.cpu_monitor.log_memory_stats("After evaluation")
                    
                    # Aggressive memory cleanup after validation
                    self._aggressive_memory_cleanup(f"After validation epoch {epoch + 1}")
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
                        if isinstance(val, dict):
                            logger.info(f"  - {key}: [nested metrics]")
                        else:
                            logger.info(f"  - {key}: {val:.4f}")

                # Clear evaluation metrics to prevent memory accumulation
                # Don't delete reid_metrics here as it's still needed for WandB logging
                if 'reid_metrics' in locals():
                    reid_metrics.clear()

                logger.info(f"Margin used: {current_margin:.4f}")

                # Log training progress metrics with diagnostic guidance
                logger.info(f"Training Progress Metrics:")
                logger.info(f"  - Embedding Variance: {epoch_embedding_variance:.4f} (target: >0.1 for diverse features)")
                logger.info(f"  - Intra-class Distance: {epoch_intra_distance:.4f} (target: <0.5 for tight clusters)")
                logger.info(f"  - Inter-class Distance: {epoch_inter_distance:.4f} (target: >1.0 for separation)")
                logger.info(f"  - Margin Satisfaction: {epoch_margin_satisfaction:.4f} (target: >0.8 for effective triplets)")
                logger.info(f"  - Hard Triplets Ratio: {epoch_hard_triplets:.4f} (target: 0.3-0.7 for balanced difficulty)")
                logger.info(f"  - Easy Triplets Ratio: {epoch_easy_triplets:.4f} (target: <0.3 to avoid trivial learning)")
                logger.info(f"  - Mining Efficiency: {epoch_mining_efficiency:.4f} (target: >0.7 for good triplet selection)")
                logger.info(f"  - Gradient Norm: {epoch_grad_norm:.4f} (target: 0.01-100, watch for explosion >1000)")
                
                # Log epoch metrics to wandb with memory management
                if wandb_logger.enabled:
                    # Create metrics dict with explicit cleanup
                    metrics = {
                        "train_loss": float(epoch_train_loss),  # Ensure scalar values
                        "margin": float(current_margin),
                        "current_lr": float(self.optimizer.param_groups[0]['lr']),
                        # Training progress metrics
                        "embedding_variance": float(epoch_embedding_variance),
                        "intra_class_distance": float(epoch_intra_distance),
                        "inter_class_distance": float(epoch_inter_distance),
                        "margin_satisfaction": float(epoch_margin_satisfaction),
                        "hard_triplets_ratio": float(epoch_hard_triplets),
                        "easy_triplets_ratio": float(epoch_easy_triplets),
                        "mining_efficiency": float(epoch_mining_efficiency),
                        "gradient_norm": float(epoch_grad_norm)
                    }
                    if epoch_val_loss is not None:
                        metrics["val_loss"] = float(epoch_val_loss)
                        # Add re-id metrics with explicit float conversion to prevent accumulation
                        for k, v in reid_metrics.items():
                            metrics[k] = float(v) if isinstance(v, (int, float)) else v
                    
                    if wandb_logger.enabled:
                        wandb_logger.log_metrics(metrics)
                    
                    # Explicitly clear metrics dict to prevent accumulation
                    del metrics

                    # Now it's safe to delete reid_metrics after WandB logging is complete
                    if 'reid_metrics' in locals():
                        del reid_metrics

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

                # Save checkpoint at the end of each epoch with aggressive memory management
                if wandb_logger.enabled and self.optimizer is not None:
                    try:
                        # Monitor memory before checkpoint save
                        import psutil
                        process = psutil.Process()
                        memory_before_checkpoint = process.memory_info().rss / 1024 / 1024  # MB
                        logger.debug(f"Memory before checkpoint save: {memory_before_checkpoint:.1f}MB")
                        
                        # Force garbage collection before checkpoint to clear accumulated state
                        gc.collect()
                        
                        model_config_dict = {
                            "margin": float(self.margin),
                            "learning_rate": float(self.learning_rate),
                            "batch_size": int(self.batch_size),
                            "weight_decay": float(self.weight_decay)
                        }
                        
                        # Memory-efficient checkpoint save: pass model/optimizer objects directly
                        # instead of pre-computed state_dicts to avoid memory leaks
                        wandb_logger.save_checkpoint(
                            epoch=epoch + 1,  # Save 1-indexed epoch number
                            model=self.model,
                            optimizer=self.optimizer,
                            loss=float(monitoring_loss),  # Ensure scalar value
                            model_name=type(self.model).__name__,
                            model_config=model_config_dict
                        )
                        logger.debug(f"Checkpoint saved for epoch {epoch + 1}")
                        
                        # Explicit cleanup after checkpoint save
                        del model_config_dict
                        gc.collect()
                        
                        # Monitor memory after checkpoint save
                        memory_after_checkpoint = process.memory_info().rss / 1024 / 1024  # MB
                        checkpoint_memory_delta = memory_after_checkpoint - memory_before_checkpoint
                        if abs(checkpoint_memory_delta) > 50:  # Lowered threshold for better monitoring
                            logger.info(f"Checkpoint memory usage: {memory_before_checkpoint:.1f}MB → {memory_after_checkpoint:.1f}MB (Δ{checkpoint_memory_delta:+.1f}MB)")
                        
                    except Exception as e:
                        if getattr(wandb_config, 'enabled', False):
                            logger.error(f"Failed to save checkpoint for epoch {epoch + 1} while wandb is enabled: {e}")
                            raise
                        logger.warning(f"Failed to save checkpoint for epoch {epoch + 1}: {e}")

                # Aggressive end-of-epoch memory cleanup
                try:
                    # Clear reid_metrics dict explicitly
                    if 'reid_metrics' in locals():
                        reid_metrics.clear()
                        del reid_metrics
                    
                    # Force garbage collection at end of each epoch
                    gc.collect()
                    
                    # Log memory after cleanup
                    if epoch % 5 == 0:  # Log every 5 epochs to monitor trend
                        import psutil
                        process = psutil.Process()
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        logger.info(f"End of epoch {epoch + 1} memory usage: {current_memory:.1f}MB")
                        
                except Exception as cleanup_error:
                    logger.debug(f"End-of-epoch cleanup warning: {cleanup_error}")

                # Additional DataLoader iterator cleanup every few epochs to prevent accumulation
                if (epoch + 1) % EPOCHS_PER_DATALOADER_RESTART == 0 and self.num_workers > 0:
                    try:
                        # Reset DataLoader iterators to prevent state accumulation
                        if hasattr(self.dataloader, '_iterator') and self.dataloader._iterator:
                            del self.dataloader._iterator
                            self.dataloader._iterator = None
                        if self.val_dataloader and hasattr(self.val_dataloader, '_iterator') and self.val_dataloader._iterator:
                            del self.val_dataloader._iterator  
                            self.val_dataloader._iterator = None
                        logger.debug(f"DataLoader iterators reset at epoch {epoch + 1}")
                    except Exception as iterator_error:
                        logger.debug(f"DataLoader iterator cleanup warning: {iterator_error}")

                # Memory leak detection and alerting
                try:
                    import psutil
                    process = psutil.Process()
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    
                    if self.epoch_memory_baseline is not None:
                        memory_increase = current_memory - self.epoch_memory_baseline
                        # Calculate per-epoch increase more accurately accounting for baseline epoch
                        epochs_since_baseline = max(1, epoch + 1 - 0)  # epoch is 0-indexed, baseline set after epoch 0 starts
                        memory_increase_per_epoch = memory_increase / epochs_since_baseline if epochs_since_baseline > 0 else 0
                        
                        # Log memory trend every epoch
                        logger.info(f"Memory trend: Baseline={self.epoch_memory_baseline:.1f}MB, Current={current_memory:.1f}MB, Increase={memory_increase:.1f}MB ({memory_increase_per_epoch:.1f}MB/epoch)")
                        
                    else:
                        # Baseline not yet established
                        logger.debug(f"Memory monitoring: {current_memory:.1f}MB (baseline not yet established)")
                        
                except Exception as memory_check_error:
                    logger.debug(f"Memory leak detection warning: {memory_check_error}")


        except InterruptedError as e:
            logger.info(f"Training cancelled by external request: {e}")
            
            # Clean up both GPU and CPU memory on cancellation
            logger.info("Cleaning up GPU and CPU memory after training cancellation...")
            
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
            self.cpu_monitor.log_memory_stats("After cancellation cleanup")
            
            # Re-raise the InterruptedError to propagate cancellation
            raise
            
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
        from evaluator import ModelEvaluator

        # dataloader may be a DataLoader; evaluator expects a Dataset instance
        dataset = getattr(dataloader, 'dataset', dataloader)

        # Create evaluator with memory management
        evaluator = None
        results = {}
        flat_metrics = {}
        
        try:
            evaluator = ModelEvaluator(self.model, device=self.device)

            # Log memory before evaluation
            import psutil
            process = psutil.Process()
            memory_before_eval = process.memory_info().rss / 1024 / 1024  # MB
            logger.debug(f"Memory before evaluation: {memory_before_eval:.1f}MB")

            try:
                # Run the comprehensive evaluation (this also saves results to disk)
                results = evaluator.evaluate_comprehensive(dataset, **self.kwargs)
                
                # Log evaluation results to WandB if enabled and results available
                if results and hasattr(evaluator, '_log_to_wandb'):
                    try:
                        evaluator._log_to_wandb(results)
                        logger.debug("Evaluation results logged to WandB")
                    except Exception as e:
                        logger.debug(f"Failed to log evaluation results to WandB: {e}")
                        
            except Exception as e:
                logger.warning(f"Comprehensive evaluation failed, falling back to local computation: {e}")
                results = {}

            # If evaluation succeeded, flatten all nested metrics into a single-level dict;
            # otherwise fall back to a local computation that produces the same metric groups.
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

        finally:
            # Always cleanup evaluator to prevent memory leaks
            if evaluator is not None:
                try:
                    evaluator.cleanup()
                except Exception as cleanup_error:
                    logger.debug(f"Evaluator cleanup warning: {cleanup_error}")
                finally:
                    del evaluator
            
            # Explicit cleanup of evaluation variables
            if 'results' in locals():
                if hasattr(results, 'clear') and callable(results.clear):
                    results.clear()
                del results
                
            # Force garbage collection after evaluation
            gc.collect()
            
            # Monitor memory after evaluation
            try:
                import psutil
                process = psutil.Process()
                memory_after_eval = process.memory_info().rss / 1024 / 1024  # MB
                eval_memory_delta = memory_after_eval - memory_before_eval
                if abs(eval_memory_delta) > 50:  # Log significant memory changes
                    logger.info(f"Evaluation memory usage: {memory_before_eval:.1f}MB → {memory_after_eval:.1f}MB (Δ{eval_memory_delta:+.1f}MB)")
            except Exception as memory_log_error:
                logger.debug(f"Memory logging warning: {memory_log_error}")

        # Additional cleanup of local variables
        try:
            del dataset, evaluator
            gc.collect()
        except NameError:
            pass

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

        # Log memory after data setup, before model setup
        if torch.cuda.is_available() and self.device.type == 'cuda':
            log_gpu_memory_stats("After data setup")

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
        
        if not wandb_logger.enabled:
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
                    # Avoid unnecessary device movement to prevent temporary memory doubling
                    if str(self.model.device) != str(self.device):
                        logger.info(f"Moving model from {self.model.device} to {self.device} after checkpoint resumption")
                        self.model.to(self.device)
                    else:
                        logger.debug(f"Model already on correct device after checkpoint resumption: {self.device}")
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

    def _smart_gpu_cache_clear(self, force: bool = False, context: str = ""):
        """
        Intelligently clear GPU cache only when necessary.
        
        Args:
            force: If True, force cache clear regardless of memory usage
            context: Context string for logging (e.g., "epoch_end", "aggressive_cleanup")
        """
        if not torch.cuda.is_available():
            return  # No GPU, no cache to clear
            
        try:
            if force:
                torch.cuda.empty_cache()
                if context:
                    logger.debug(f"GPU cache force-cleared ({context})")
                return
                
            # Check if cache clearing is needed based on memory utilization
            if hasattr(torch.cuda, 'memory_reserved') and torch.cuda.memory_reserved() > 0:
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                gpu_utilization = memory_allocated / memory_reserved
                
                # Use configurable threshold (default 85% for GPU Cloud Run)
                threshold = getattr(self, 'gpu_cache_threshold', 0.85)
                
                if gpu_utilization > threshold:
                    torch.cuda.empty_cache()
                    logger.debug(f"GPU cache cleared ({context}) - utilization: {gpu_utilization:.1%} > {threshold:.1%}")
                else:
                    logger.debug(f"GPU cache clear skipped ({context}) - utilization: {gpu_utilization:.1%} <= {threshold:.1%}")
            else:
                # No memory allocated yet, no need to clear
                logger.debug(f"GPU cache clear skipped ({context}) - no memory allocated")
                
        except Exception as e:
            # Conservative fallback for Cloud Run GPU: only clear on high memory conditions
            if getattr(self, 'conservative_gpu_cache', True):
                logger.debug(f"GPU cache clear skipped ({context}) - conservative mode, error: {e}")
            else:
                # Fallback: clear cache if memory queries fail and not in conservative mode
                torch.cuda.empty_cache()
                logger.debug(f"GPU cache cleared ({context}) - fallback due to error: {e}")

    def cleanup_model(self):
        """
        Clean up the model and associated resources to free memory.
        
        This method should be called after evaluation is complete to prevent
        memory leaks from accumulated model references.
        """
        if hasattr(self, 'model') and self.model is not None:
            logger.info("Cleaning up training model and resources")
            
            # Clear model from GPU memory with smart cache management
            if torch.cuda.is_available():
                self.model = self.model.cpu()
                # Use smart cache clearing instead of aggressive clearing
                self._smart_gpu_cache_clear(force=True, context="model_cleanup")
            
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

    def _aggressive_memory_cleanup(self, context: str = ""):
        """
        Perform aggressive memory cleanup to prevent accumulation across epochs.
        
        Args:
            context: Context string for logging
        """
        try:
            import psutil

            # Log memory before cleanup
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            # Smart GPU cache clearing - force clear for aggressive cleanup
            if torch.cuda.is_available():
                self._smart_gpu_cache_clear(force=True, context=f"aggressive_cleanup_{context}")
                
            # Clear CPU memory
            clear_cpu_memory(force=True)
            
            # Additional PyTorch internal cleanup
            if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCaches'):
                try:
                    torch._C._cuda_clearCaches()  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # Log memory after cleanup
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before
            
            if abs(memory_delta) > 10:  # Log if significant change
                logger.info(f"Aggressive memory cleanup {context}: {memory_before:.1f}MB → {memory_after:.1f}MB (Δ{memory_delta:+.1f}MB)")
            else:
                logger.debug(f"Aggressive memory cleanup {context}: {memory_before:.1f}MB → {memory_after:.1f}MB (Δ{memory_delta:+.1f}MB)")
                
        except Exception as e:
            logger.debug(f"Aggressive memory cleanup failed {context}: {e}")


