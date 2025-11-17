import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)
from services.service_training.src.evaluator import (ModelEvaluator,
                                                     calculate_embedding_variance,
                                                     calculate_gradient_norm,
                                                     calculate_intra_inter_distances,
                                                     calculate_triplet_mining_efficiency)
from torch.utils.data import DataLoader, Dataset
from services.service_training.src.wandb_logger import wandb_logger

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
EPOCHS_PER_DATALOADER_RESTART = 50
THRESHOLD_FOR_DATALOADER_RESTART = 90.0  # Memory usage percentage threshold


@dataclass
class TrainingMetrics:
    """Container for training metrics accumulated during an epoch."""
    
    loss: float = 0.0
    embedding_variance: float = 0.0
    intra_distance: float = 0.0
    inter_distance: float = 0.0
    margin_satisfaction: float = 0.0
    hard_triplets: float = 0.0
    easy_triplets: float = 0.0
    mining_efficiency: float = 0.0
    grad_norm: float = 0.0
    
    def accumulate(self, loss: float, embedding_variance: float, 
                   distance_metrics: Dict[str, float], mining_metrics: Dict[str, float],
                   grad_norm: float) -> None:
        """Accumulate metrics from a single batch."""
        self.loss += loss
        self.embedding_variance += embedding_variance
        self.intra_distance += distance_metrics['intra_class_distance']
        self.inter_distance += distance_metrics['inter_class_distance']
        self.margin_satisfaction += distance_metrics['triplet_margin_satisfaction']
        self.hard_triplets += mining_metrics['hard_triplets_ratio']
        self.easy_triplets += mining_metrics['easy_triplets_ratio']
        self.mining_efficiency += mining_metrics['mining_efficiency']
        self.grad_norm += grad_norm
    
    def compute_averages(self, num_batches: int) -> 'TrainingMetrics':
        """Compute average metrics over all batches."""
        if num_batches == 0:
            return TrainingMetrics()
        
        return TrainingMetrics(
            loss=self.loss / num_batches,
            embedding_variance=self.embedding_variance / num_batches,
            intra_distance=self.intra_distance / num_batches,
            inter_distance=self.inter_distance / num_batches,
            margin_satisfaction=self.margin_satisfaction / num_batches,
            hard_triplets=self.hard_triplets / num_batches,
            easy_triplets=self.easy_triplets / num_batches,
            mining_efficiency=self.mining_efficiency / num_batches,
            grad_norm=self.grad_norm / num_batches
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for logging."""
        return {
            'train_loss': self.loss,
            'embedding_variance': self.embedding_variance,
            'intra_class_distance': self.intra_distance,
            'inter_class_distance': self.inter_distance,
            'margin_satisfaction': self.margin_satisfaction,
            'hard_triplets_ratio': self.hard_triplets,
            'easy_triplets_ratio': self.easy_triplets,
            'mining_efficiency': self.mining_efficiency,
            'gradient_norm': self.grad_norm
        }
    
    def log_summary(self, epoch: int, margin: float) -> None:
        """Log training metrics summary with diagnostic guidance."""
        logger.info(f"=== Epoch {epoch} Training Summary ===")
        logger.info(f"Training Loss: {self.loss:.4f}")
        logger.info(f"Margin used: {margin:.4f}")
        logger.info(f"Training Progress Metrics:")
        logger.info(f"  - Embedding Variance: {self.embedding_variance:.4f} (target: >0.1 for diverse features)")
        logger.info(f"  - Intra-class Distance: {self.intra_distance:.4f} (target: <0.5 for tight clusters)")
        logger.info(f"  - Inter-class Distance: {self.inter_distance:.4f} (target: >1.0 for separation)")
        logger.info(f"  - Margin Satisfaction: {self.margin_satisfaction:.4f} (target: >0.8 for effective triplets)")
        logger.info(f"  - Hard Triplets Ratio: {self.hard_triplets:.4f} (target: 0.3-0.7 for balanced difficulty)")
        logger.info(f"  - Easy Triplets Ratio: {self.easy_triplets:.4f} (target: <0.3 to avoid trivial learning)")
        logger.info(f"  - Mining Efficiency: {self.mining_efficiency:.4f} (target: >0.7 for good triplet selection)")
        logger.info(f"  - Gradient Norm: {self.grad_norm:.4f} (target: 0.01-100, watch for explosion >1000)")


class MemoryManager:
    """Unified memory management for training with smart cleanup strategies."""
    
    def __init__(self, device: torch.device, gpu_cache_threshold: float = 0.85, 
                 conservative_mode: bool = True):
        """
        Initialize memory manager.
        
        Args:
            device: The device being used for training
            gpu_cache_threshold: GPU utilization threshold for cache clearing (default: 85%)
            conservative_mode: If True, only clear GPU cache when necessary
        """
        self.device = device
        self.gpu_cache_threshold = gpu_cache_threshold
        self.conservative_mode = conservative_mode
        self.baseline_memory: Optional[float] = None
        
    def establish_baseline(self) -> None:
        """Establish memory baseline for leak detection."""
        try:
            import psutil
            process = psutil.Process()
            self.baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory baseline established: {self.baseline_memory:.1f}MB")
        except Exception as e:
            logger.debug(f"Could not establish memory baseline: {e}")
    
    def smart_gpu_cache_clear(self, force: bool = False, context: str = "") -> None:
        """
        Intelligently clear GPU cache only when necessary.
        
        Args:
            force: If True, force cache clear regardless of memory usage
            context: Context string for logging
        """
        if not torch.cuda.is_available():
            return
            
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
                
                if gpu_utilization > self.gpu_cache_threshold:
                    torch.cuda.empty_cache()
                    logger.debug(f"GPU cache cleared ({context}) - utilization: {gpu_utilization:.1%} > {self.gpu_cache_threshold:.1%}")
                else:
                    logger.debug(f"GPU cache clear skipped ({context}) - utilization: {gpu_utilization:.1%} <= {self.gpu_cache_threshold:.1%}")
            else:
                logger.debug(f"GPU cache clear skipped ({context}) - no memory allocated")
                
        except Exception as e:
            if self.conservative_mode:
                logger.debug(f"GPU cache clear skipped ({context}) - conservative mode, error: {e}")
            else:
                torch.cuda.empty_cache()
                logger.debug(f"GPU cache cleared ({context}) - fallback due to error: {e}")
    
    def aggressive_cleanup(self, context: str = "") -> None:
        """
        Perform aggressive memory cleanup to prevent accumulation.
        
        Args:
            context: Context string for logging
        """
        try:
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            # Force GPU cache clear for aggressive cleanup
            if torch.cuda.is_available():
                self.smart_gpu_cache_clear(force=True, context=f"aggressive_{context}")
                
            # Clear CPU memory
            clear_cpu_memory(force=True)
            
            # Additional PyTorch internal cleanup
            if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCaches'):
                try:
                    torch._C._cuda_clearCaches()  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # Log memory change
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_after - memory_before
            
            if abs(memory_delta) > 10:
                logger.info(f"Aggressive cleanup {context}: {memory_before:.1f}MB → {memory_after:.1f}MB (Δ{memory_delta:+.1f}MB)")
            else:
                logger.debug(f"Aggressive cleanup {context}: {memory_before:.1f}MB → {memory_after:.1f}MB (Δ{memory_delta:+.1f}MB)")
                
        except Exception as e:
            logger.debug(f"Aggressive cleanup failed {context}: {e}")
    
    def periodic_cleanup(self, epoch: int, cleanup_interval: int = 2) -> None:
        """
        Perform periodic cleanup to prevent memory accumulation.
        
        Args:
            epoch: Current epoch number (0-indexed)
            cleanup_interval: How often to run cleanup (every N epochs)
        """
        if (epoch + 1) % cleanup_interval == 0:
            try:
                self.aggressive_cleanup(f"epoch_{epoch + 1}")
            except Exception as e:
                logger.debug(f"Periodic cleanup warning: {e}")
    
    def check_for_leaks(self, epoch: int) -> None:
        """
        Check for memory leaks by comparing current usage to baseline.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        if self.baseline_memory is None:
            return
            
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = current_memory - self.baseline_memory
            epochs_since_baseline = max(1, epoch + 1)
            memory_increase_per_epoch = memory_increase / epochs_since_baseline
            
            logger.info(f"Memory trend: Baseline={self.baseline_memory:.1f}MB, "
                       f"Current={current_memory:.1f}MB, "
                       f"Increase={memory_increase:.1f}MB ({memory_increase_per_epoch:.1f}MB/epoch)")
            
        except Exception as e:
            logger.debug(f"Memory leak detection warning: {e}")
    
    def cleanup_on_error(self, error_type: str = "unknown") -> None:
        """
        Clean up memory after an error occurs.
        
        Args:
            error_type: Type of error that occurred
        """
        logger.info(f"Cleaning up memory after {error_type} error...")
        
        # Force aggressive cleanup
        clear_gpu_memory(force=True) if torch.cuda.is_available() else None
        clear_cpu_memory(force=True)
        
        if torch.cuda.is_available():
            log_gpu_memory_stats(f"After {error_type} cleanup")
    
    def log_status(self, context: str = "") -> None:
        """Log current memory status."""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            log_gpu_memory_stats(context)
        
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.debug(f"CPU memory {context}: {current_memory:.1f}MB")
        except Exception:
            pass


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
            **kwargs: Training parameters (num_epochs, batch_size, learning_rate, etc.)
        
    All hyperparameters default to values from training_config/wandb_config 
    but can be overridden via kwargs.
        """
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

        # Initialize training parameters - direct fallback from kwargs to config
        self.num_epochs = kwargs.get('num_epochs', training_config.num_epochs)
        self.batch_size = kwargs.get('batch_size', training_config.batch_size)
        self.learning_rate = kwargs.get('learning_rate', training_config.learning_rate)
        self.margin = kwargs.get('margin', training_config.margin)
        self.weight_decay = kwargs.get('weight_decay', training_config.weight_decay)
        self.lr_scheduler_patience = kwargs.get('lr_scheduler_patience', training_config.lr_scheduler_patience)
        self.lr_scheduler_threshold = kwargs.get('lr_scheduler_threshold', training_config.lr_scheduler_threshold)
        self.lr_scheduler_min_lr = kwargs.get('lr_scheduler_min_lr', training_config.lr_scheduler_min_lr)
        self.lr_scheduler_factor = kwargs.get('lr_scheduler_factor', training_config.lr_scheduler_factor)
        self.force_pretraining = kwargs.get('force_pretraining', training_config.force_pretraining)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', training_config.early_stopping_patience)
        self.min_images_per_player = kwargs.get('min_images_per_player', training_config.min_images_per_player)
        self.train_prefetch_factor = kwargs.get('train_prefetch_factor', training_config.prefetch_factor)
        self.margin_decay_rate = kwargs.get('margin_decay_rate', training_config.margin_decay_rate)
        self.margin_change_threshold = kwargs.get('margin_change_threshold', training_config.margin_change_threshold)
        self.train_ratio = kwargs.get('train_ratio', training_config.train_ratio)
        self.n_datasets_to_use = kwargs.get('n_datasets_to_use', training_config.n_datasets_to_use)
        self.dataset_address = kwargs.get('dataset_address', training_config.dataset_address)
        self.wandb_project = kwargs.get('wandb_project', wandb_config.project)
        
        # Backward compatibility aliases
        self.scheduler_patience = self.lr_scheduler_patience
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
        
        # Initialize unified memory manager
        gpu_cache_threshold = kwargs.get('gpu_cache_threshold', 0.85)
        conservative_mode = kwargs.get('conservative_gpu_cache', True)
        self.memory_manager = MemoryManager(
            device=self.device,
            gpu_cache_threshold=gpu_cache_threshold,
            conservative_mode=conservative_mode
        )


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
                "optimizer_type": "AdamW",
                "model_architecture": type(self.model).__name__ if self.model is not None else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
                    logger.info("Enabled backbone fine-tuning ")
                except Exception as e:
                    logger.warning(f"Could not enable backbone fine-tuning: {e}")
            else:
                logger.debug("enable_backbone_fine_tuning is not callable or not present on model")
            
            # Setup training components with improved loss function
            # Use TripletMarginLoss with distance weighting for better convergence
            self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2, reduction='mean')
            
            # Setup classification loss if the model has a classification head
            self.classification_loss_fn = None
            if hasattr(self.model, 'classification_head') and self.model.classification_head is not None:
                self.classification_loss_fn = nn.CrossEntropyLoss()
                logger.info("Classification head detected - will use combined loss for early epochs")
            
            def _collect_params(module):
                params: List[torch.nn.Parameter] = []
                if module is None:
                    return params
                for param in module.parameters():
                    if id(param) not in included_param_ids:
                        included_param_ids.add(id(param))
                        params.append(param)
                return params

            def _resolve_component(*attr_paths):
                if self.model is None:
                    return None
                for path in attr_paths:
                    module = self.model
                    for attr in path:
                        module = getattr(module, attr, None)
                        if module is None:
                            break
                    else:
                        return module
                return None

            included_param_ids: set[int] = set()
            param_groups: List[Dict[str, Any]] = []
            component_specs = [
                ("layer3", 2e-5, (("layer3",), ("backbone", "layer3"))),
                ("layer4", 5e-5, (("layer4",), ("backbone", "layer4"))),
                ("cbam", 1e-4, (("cbam",), ("backbone", "cbam"))),
                ("head", 2e-4, (("head",), ("backbone", "head"), ("backbone", "_head"))),
            ]

            added_components: List[str] = []
            for name, lr, paths in component_specs:
                module = _resolve_component(*paths)
                params = _collect_params(module)
                if params:
                    param_groups.append({"params": params, "lr": lr})
                    added_components.append(name)

            if getattr(self.model, "classification_head", None) is not None:
                params = _collect_params(self.model.classification_head)
                if params:
                    param_groups.append({"params": params, "lr": max(self.learning_rate, 1e-3)})
                    added_components.append("classification_head")

            remaining_params = [
                param
                for param in self.model.parameters()
                if id(param) not in included_param_ids
            ]
            if remaining_params:
                param_groups.append({"params": remaining_params, "lr": self.learning_rate})
                added_components.append("remaining_params")

            if not param_groups:
                raise RuntimeError("Failed to collect any parameter groups for optimizer setup")

            # Use AdamW optimizer which is better for fine-tuning transformers
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),  # Default AdamW betas
                eps=1e-8  # Slightly higher epsilon for stability
            )
            logger.info(f"Optimizer parameter groups: {', '.join(added_components)}")
            log_gpu_memory_stats("After creating optimizer")
            # Scheduler uses its own patience/threshold for LR adjustment only
        
            # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     self.optimizer,
            #     mode='min',
            #     factor=self.lr_scheduler_factor,
            #     patience=self.scheduler_patience,
            #     threshold=self.scheduler_threshold,
            #     min_lr=self.lr_scheduler_min_lr
            # )


            warmup_epochs = 5
            total_epochs = self.num_epochs

            self.lr_scheduler = SequentialLR(
                self.optimizer,
                schedulers=[
                    LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                    CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7)
                ],
                milestones=[warmup_epochs]
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
            logger.info(f"Optimizer: AdamW (initial lr={self.learning_rate}, weight_decay={self.weight_decay})")
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

                # Periodic memory cleanup to prevent accumulation
                self.memory_manager.periodic_cleanup(epoch, cleanup_interval=2)
                
                self.memory_manager.log_status(f"Epoch {epoch + 1} start")
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
                ttl_batches = len(self.dataloader)

                # Initialize metrics tracker for this epoch
                epoch_metrics = TrainingMetrics()

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
                    
                for i, (anchor, positive, negative, labels) in enumerate(self.dataloader):

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
                    labels = labels.to(self.device, non_blocking=True)

                    # Clear gradients
                    self.optimizer.zero_grad()
                    
                    # Determine if we should use classification loss
                    # Use combined loss for first N epochs to jump-start embeddings
                    use_classification = (self.classification_loss_fn is not None and 
                                        epoch < training_config.classification_epochs)
                    
                    # Forward pass
                    if use_classification:
                        # Get embeddings and logits for classification
                        (emb_anchor, logits_a), (emb_positive, logits_p), (emb_negative, logits_n) = \
                            self.model.forward_triplet(anchor, positive, negative, return_logits=True)  # pyright: ignore[reportCallIssue]
                        
                        # Compute triplet loss
                        triplet_loss = self.loss_fn(emb_anchor, emb_positive, emb_negative)
                        
                        # Compute classification loss
                        # Concatenate logits and labels for batch processing
                        all_logits = torch.cat([logits_a, logits_p, logits_n], dim=0)
                        all_labels = labels.repeat(3)  # Repeat labels for anchor, positive, negative
                        classification_loss = self.classification_loss_fn(all_logits, all_labels)
                        
                        # Combined loss with decreasing weight for classification
                        # Start with configured weight, linearly decrease to 0 over N epochs
                        progress = epoch / training_config.classification_epochs
                        class_weight = max(0.0, training_config.classification_weight_start * (1.0 - progress))
                        loss = triplet_loss + (class_weight * classification_loss)
                        
                        # Log the component losses periodically
                        if (i + 1) % BATCHES_PER_LOG_MSG == 0:
                            logger.debug(f"Epoch {epoch+1} Batch {i+1}: triplet_loss={triplet_loss.item():.4f}, "
                                       f"class_loss={classification_loss.item():.4f}, class_weight={class_weight:.2f}")
                    else:
                        # Standard triplet-only training
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
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # Extract scalar immediately and accumulate metrics
                    loss_value = loss.item()
                    epoch_metrics.accumulate(
                        loss=loss_value,
                        embedding_variance=embedding_variance,
                        distance_metrics=distance_metrics,
                        mining_metrics=mining_metrics,
                        grad_norm=grad_norm
                    )
                    
                    # Explicitly delete ALL tensors and intermediate results to prevent accumulation
                    del anchor, positive, negative, emb_anchor, emb_positive, emb_negative, loss
                    del concatenated_embeddings  # Clean up the concatenated tensor
                    
                    # Establish memory baseline after first few batches of first epoch
                    if epoch == 0 and i == 10 and self.memory_manager.baseline_memory is None:
                        self.memory_manager.establish_baseline()
                    

                # Calculate average metrics for the epoch
                avg_metrics = epoch_metrics.compute_averages(ttl_batches)
                

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
                                self.memory_manager.smart_gpu_cache_clear(force=False, context=f"validation_batch_{j+1}")

                    epoch_val_loss = running_val_loss / ttl_batches if ttl_batches > 0 else 0.0

                    # 2. Calculate Retrieval Metrics with memory monitoring
                    self.memory_manager.log_status("Before evaluation")
                    self.cpu_monitor.log_memory_stats("Before evaluation")
                    
                    reid_metrics = self._evaluate_reid_metrics(val_dataloader)
                    
                    self.memory_manager.log_status("After evaluation")
                    self.cpu_monitor.log_memory_stats("After evaluation")
                    
                    # Aggressive memory cleanup after validation
                    self.memory_manager.aggressive_cleanup(f"After validation epoch {epoch + 1}")
                    self.cpu_monitor.log_memory_stats(f"Epoch {epoch + 1} validation complete")

                # ========================================================================
                # Log and check for early stopping
                # ========================================================================
                # Use validation loss for monitoring if available, otherwise fall back to training loss
                monitoring_loss = avg_metrics.loss
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

                # Log training metrics summary
                avg_metrics.log_summary(epoch + 1, current_margin)
                
                # Log epoch metrics to wandb with memory management
                if wandb_logger.enabled:
                    # Create metrics dict from TrainingMetrics
                    metrics = avg_metrics.to_dict()
                    metrics["margin"] = float(current_margin)
                    metrics["current_lr"] = float(self.optimizer.param_groups[0]['lr'])
                    
                    if epoch_val_loss is not None:
                        metrics["val_loss"] = float(epoch_val_loss)
                        # Add re-id metrics with explicit float conversion to prevent accumulation
                        for k, v in reid_metrics.items():
                            metrics[k] = float(v) if isinstance(v, (int, float)) else v
                    
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
                    if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                        self.lr_scheduler.step(monitoring_loss)
                    else:
                        self.lr_scheduler.step()
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
                            logger.debug(f"Checkpoint memory usage: {memory_before_checkpoint:.1f}MB → {memory_after_checkpoint:.1f}MB (Δ{checkpoint_memory_delta:+.1f}MB)")
                        
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
                self.memory_manager.check_for_leaks(epoch)


        except InterruptedError as e:
            logger.info(f"Training cancelled by external request: {e}")
            self.memory_manager.cleanup_on_error("cancellation")
            self.cpu_monitor.log_memory_stats("After cancellation cleanup")
            raise
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.error(f"Training failed with error: {e}")
            self.memory_manager.cleanup_on_error("runtime")
            self.cpu_monitor.log_memory_stats("After failure cleanup")
            raise
            
        except Exception as e:
            logger.error(f"Training failed with unexpected error: {e}")
            
            # Check if this is a CUDA OOM error and provide specific guidance
            if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                logger.error("CUDA Out of Memory Error detected!")
                self.memory_manager.log_status("Before OOM cleanup")
                self.memory_manager.cleanup_on_error("OOM")
                self.cpu_monitor.log_memory_stats("After OOM cleanup")
                logger.error("Suggestions to resolve OOM:")
                logger.error("1. Reduce batch_size in your configuration")
                logger.error("2. Use gradient accumulation to simulate larger batches")
                logger.error("3. Consider mixed precision training (fp16)")
                logger.error("4. Use a smaller model architecture")
            else:
                self.memory_manager.cleanup_on_error("unexpected")
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
        from services.service_training.src.evaluator import ModelEvaluator

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
                    logger.debug(f"Evaluation memory usage: {memory_before_eval:.1f}MB → {memory_after_eval:.1f}MB (Δ{eval_memory_delta:+.1f}MB)")
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

        # Add num_classes to model_kwargs if dataset has player information and classification head is enabled
        if training_config.use_classification_head and 'num_classes' not in model_kwargs and hasattr(dataset, 'players'):
            num_classes = len(dataset.players)
            model_kwargs['num_classes'] = num_classes
            logger.info(f"Setting num_classes={num_classes} for classification head (enabled for {training_config.classification_epochs} epochs)")

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
                self.memory_manager.smart_gpu_cache_clear(force=True, context="model_cleanup")
            
            # Clear optimizer state
            if hasattr(self, 'optimizer') and self.optimizer is not None:
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


