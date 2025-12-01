import logging
logger = logging.getLogger(__name__)

import os
import random

import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

from shared_libs.common.google_storage import GCSPaths, get_storage
from shared_libs.config.transforms import get_transforms
from shared_libs.utils.id_generator import create_simple_uuid


from schemas.training import TrainingParams, EvalParams
from wandb_logger import wandb_logger
from training_loop import TrainingLoop
from loss_fn import loss_fn
from model import ReIdModel
from eval_dataset import load_eval_dataset
from dataset import LacrossePlayerDataset



class TrainingController():

    def __init__(
            self, 
            tenant_id: str, 
            wandb_run_name: str,
            training_params: TrainingParams,
            eval_params: EvalParams = EvalParams(),
            task_id: Optional[str] = None
        ):
        """
        Sets up all components required for training.
        """

        self.tenant_id = tenant_id
        self.wandb_run_name = wandb_run_name
        self.wandb_run_id = None  # Initialize run ID
        self.task_id = task_id

        self.storage_client = get_storage(tenant_id)
        self.path_manager = GCSPaths()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.training_params = training_params
        self.eval_params = eval_params

        self.wandb_logger = wandb_logger
        self.loss_fn = loss_fn
        
        # Initialize to None before loading
        self.model = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.checkpoint_data = None
        
        self.margin = self.training_params.margin
        self.starting_epoch = 0
        
        self.load_model_and_datasets()

        # Optimizer and scheduler depend on model parameters
        self.optimizer = AdamW(self.model.parameters(), lr=self.training_params.lr_initial)
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=self.training_params.lr_scheduler_factor,
            patience=self.training_params.lr_scheduler_patience,
        )

        # Load optimizer and scheduler states if available from checkpoint
        if self.checkpoint_data:
            if 'optimizer_state_dict' in self.checkpoint_data:
                self.optimizer.load_state_dict(self.checkpoint_data['optimizer_state_dict'])
                logger.info("Restored optimizer state from checkpoint")
            else:
                logger.warning("No optimizer state found in checkpoint")
                
            if 'lr_scheduler_state_dict' in self.checkpoint_data:
                self.lr_scheduler.load_state_dict(self.checkpoint_data['lr_scheduler_state_dict'])
                logger.info("Restored LR scheduler state from checkpoint")
            else:
                logger.warning("No LR scheduler state found in checkpoint")

        self.setup_wandb_logger(run_name=self.wandb_run_name)
        
        self.training_loop = TrainingLoop(
            model=self.model,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            margin=self.margin,
            loss_fn=self.loss_fn,
            optimizer = self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=self.device,
            starting_epoch=self.starting_epoch,
            num_epochs=self.training_params.num_epochs,
            # training_params is not a parameter of TrainingLoop
            wandb_logger=self.wandb_logger
        )

        self._is_running = False
        self.cancellation_requested_flag = False

        self.log_initialization_parameters()

    def train(self) -> str:
        """Run the training process synchronously.

        Returns:
            task_id: Simple UUID identifying this training run.
        """
        task_id = self.task_id or create_simple_uuid()
        logger.info(f"Starting training task {task_id} (blocking) with WandB run '{self.wandb_run_name}'")

        def _cancellation_requested() -> bool:
            return self.cancellation_requested_flag

        self._is_running = True
        try:
            self.training_loop.train(task_id, cancellation_requested_fn=_cancellation_requested)
            self.wandb_logger.finish()
        except Exception as e:
            logger.exception(f"Training task {task_id} failed: {e}")
            raise
        finally:
            self._is_running = False
        
        return task_id

    def graceful_cancellation_request(self, task_id: str, timout: int = 600) -> bool:
        """Requests graceful cancellation."""
        logger.info(f"Graceful cancellation requested for training task {task_id}")
        self.cancellation_requested_flag = True
        return True

    def train_is_running(self, task_id: str) -> bool:
        """Check if the training task is still running."""
        return self._is_running


    def setup_wandb_logger(self, run_name: str):
        """Initialize WandB logger."""
        config = {
            "training_params": self.training_params.model_dump(),
            "eval_params": self.eval_params.model_dump(),
            "task_id": self.task_id,
            "tenant_id": self.tenant_id
        }
        
        # Add components if they are already initialized
        if self.model:
            config["model_class"] = self.model.__class__.__name__
        if self.loss_fn:
            config["loss_function"] = self.loss_fn.__class__.__name__
        if hasattr(self, 'optimizer') and self.optimizer:
            config["optimizer"] = self.optimizer.__class__.__name__
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler:
            config["lr_scheduler"] = self.lr_scheduler.__class__.__name__
        if hasattr(self, 'margin') and self.margin:
            config["margin"] = self.margin
            
        # Pass wandb_run_id if available to resume the run
        self.wandb_logger.init_run(
            config=config, 
            run_name=run_name,
            run_id=self.wandb_run_id
        )

    def update_wandb_config(self):
        """Update WandB config with model details after initialization."""
        # We access the underlying wandb run object if available
        if hasattr(self.wandb_logger, 'run') and self.wandb_logger.run:
            config_update = {}
            if self.model:
                config_update["model_class"] = self.model.__class__.__name__
            if hasattr(self, 'optimizer') and self.optimizer:
                config_update["optimizer"] = self.optimizer.__class__.__name__
            if hasattr(self, 'lr_scheduler') and self.lr_scheduler:
                config_update["lr_scheduler"] = self.lr_scheduler.__class__.__name__
            if hasattr(self, 'margin') and self.margin:
                config_update["margin"] = self.margin
            
            self.wandb_logger.run.config.update(config_update, allow_val_change=True)
    
    def prepare_model(self) -> Tuple[torch.nn.Module, Optional[Dict[str, Any]]]:
        """Setup the model class and training/evaluation parameters."""
        weights_source = self.training_params.weights
        model = None
        checkpoint_data = None

        try: 
            model_name = ReIdModel.model_name
            
            if weights_source == "checkpoint":
                logger.info("Attempting to load model weights from checkpoint")
                logger.info(f"Current WandB run name: {self.wandb_run_name}")
                # load_checkpoint returns Optional[StateDicts]
                # Pass run_name explicitly since WandB run isn't initialized yet
                checkpoint_data = self.wandb_logger.load_checkpoint(run_name=self.wandb_run_name)
                
                if checkpoint_data:
                    logger.info(f"✅ Checkpoint loaded successfully")
                    logger.info(f"Checkpoint contains epoch: {checkpoint_data.get('epoch', 'NOT_FOUND')}")
                    
                    # Extract WandB Run ID if present
                    if 'wandb_run_id' in checkpoint_data:
                        self.wandb_run_id = checkpoint_data['wandb_run_id']
                        logger.info(f"Found WandB Run ID in checkpoint: {self.wandb_run_id}")

                    model = ReIdModel(pretrained=False)
                    # Load model state
                    model.load_state_dict(checkpoint_data['model_state_dict'])
                    
                    # Set starting epoch
                    # Try explicit 'epoch' key first (new format), then scheduler 'last_epoch' (fallback), then 0
                    saved_epoch = checkpoint_data.get('epoch')
                    if saved_epoch is not None:
                        self.starting_epoch = saved_epoch + 1
                    else:
                        # Fallback for old checkpoints
                        saved_epoch = checkpoint_data['lr_scheduler_state_dict'].get('last_epoch')
                        self.starting_epoch = saved_epoch + 1 if saved_epoch else 0
                   
                    logger.info(f"✅ Resuming training from epoch {self.starting_epoch}")
                else:
                    logger.warning(f"❌ No checkpoint found for run {self.wandb_run_name}, falling back to 'latest'")
                    weights_source = "latest"
            
            if weights_source == "latest":
                logger.info("Attempting to load weights from latest saved model")
                model = self.wandb_logger.load_model_from_registry(
                    model_class=ReIdModel,
                    collection_name=model_name,
                    alias="latest",
                    device=str(self.device),
                    pretrained=False
                )
                
                if not model:
                    logger.warning("No latest saved model found, falling back to 'reset'")
                    weights_source = "reset"
            
            if weights_source == "reset" or model is None:
                logger.warning("⚠️  Resetting model weights to ResNet defaults as per training parameters")
                model = ReIdModel()

            model.to(self.device)
            return model, checkpoint_data

        except Exception as e:
            msg = f"Error setting up model with weights source '{weights_source}': {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
    
    def _normalize_dataset_address(self, address: str) -> str:
        """Strip bucket name and tenant prefix from dataset address.
        
        GoogleStorageClient automatically adds tenant prefix, so we need
        to provide paths relative to the tenant root.
        
        Examples:
            laxai_dev/tenant1/datasets/foo -> datasets/foo
            tenant1/datasets/foo -> datasets/foo
            datasets/foo -> datasets/foo (already correct)
        """
        original_address = address
        
        # Strategy 0: Robust extraction from "datasets/" onward
        # This handles cases where bucket name or tenant prefix might vary or have typos
        if '/datasets/' in address:
            datasets_index = address.index('/datasets/')
            address = address[datasets_index + 1:]  # +1 to skip the leading "/"
            logger.info(f"Normalized by extracting '/datasets/': {original_address} -> {address}")
            return address
        
        # Get bucket name from storage client
        bucket_name = self.storage_client.bucket_name
        
        # Strategy 1: Try to strip "bucket_name/tenant_id/" prefix
        if bucket_name and self.tenant_id:
            combined_prefix = f"{bucket_name}/{self.tenant_id}/"
            if address.startswith(combined_prefix):
                address = address[len(combined_prefix):]
                logger.info(f"Stripped combined prefix '{combined_prefix}': {original_address} -> {address}")
                return address
        
        # Strategy 2: Try to strip just bucket name
        if bucket_name and address.startswith(f"{bucket_name}/"):
            address = address[len(bucket_name) + 1:]
            logger.info(f"Stripped bucket prefix '{bucket_name}/': {original_address} -> {address}")
        
        # Strategy 3: Try to strip just tenant prefix
        if address.startswith(f"{self.tenant_id}/"):
            address = address[len(self.tenant_id) + 1:]
            logger.info(f"Stripped tenant prefix '{self.tenant_id}/': {original_address} -> {address}")
        
        if address == original_address:
            logger.info(f"Dataset address unchanged (already normalized): {address}")
        
        return address

    def prepare_train_dataset(self) -> Dataset:
        """Prepare training dataset with triple (anchor, positive, negative) sampling."""
        try:
            train_transform = get_transforms('training')
            dataset_addresses = self.training_params.dataset_address
            if type(dataset_addresses) is str:
                dataset_addresses = [dataset_addresses]
            
            # Normalize addresses to strip bucket/tenant prefixes
            normalized_addresses = [self._normalize_dataset_address(addr) for addr in dataset_addresses]
            
            logger.info(" ")
            logger.info("="*60)
            logger.info("📦 TRAINING DATASET CONFIGURATION")
            logger.info("="*60)
            logger.info(f"Original addresses: {dataset_addresses}")
            logger.info(f"Normalized addresses: {normalized_addresses}")
            adjusted_addresses = [addr.rstrip("/") + "/train/" for addr in normalized_addresses]
            logger.info("Final paths to query for players:")
            for addr in adjusted_addresses:
                logger.info(f" - {addr}")

            train_dataset = LacrossePlayerDataset(
                storage_client=self.storage_client,
                image_dir=adjusted_addresses,
                transform=train_transform,
            )
            return train_dataset
        except Exception as e:
            msg = f"Error preparing training dataset: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def prepare_eval_dataset(self) -> Dataset:
        """
        Prepare evaluation dataset. Differently from the training dataset,
        there is no triple sampling. Only the largest val dataset is used in order
        to prevent player overlap between two different val sets (each player is 
        expected to be unique in the eval dataset).
        """
        try:
            eval_transform = get_transforms('validation')
            dataset_addresses = self.training_params.dataset_address
            if type(dataset_addresses) is str:
                dataset_addresses = [dataset_addresses]
            
            # Normalize addresses to strip bucket/tenant prefixes
            normalized_addresses = [self._normalize_dataset_address(addr) for addr in dataset_addresses]
            
            largest_dataset_path: str = ""
            largest_size: int = 0
            largest_dataset_images_paths = set()
            for dataset_address in normalized_addresses:
                dataset_address = dataset_address.rstrip("/") + "/val/"
                image_set = self.storage_client.list_blobs(dataset_address)
                size = len(image_set)
                if size > largest_size:
                    largest_size = size
                    largest_dataset_path = dataset_address
                    largest_dataset_images_paths = image_set
            logger.info(f"Evaluation dataset built from {largest_dataset_path} with {largest_size} images")
            eval_dataset = load_eval_dataset(
                storage_client=self.storage_client,
                image_paths=largest_dataset_images_paths,
                transform=eval_transform
            )
            return eval_dataset
        except Exception as e:
            msg = f"Error preparing evaluation dataset: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
    
    def prepare_train_dataloader(self) -> DataLoader:
        """Create DataLoader for training dataset."""
        batch_size = self.training_params.batch_size
        num_workers = self.training_params.num_workers
        shuffle = True
        train_dataset = self.prepare_train_dataset()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        return train_dataloader
    
    def prepare_eval_dataloader(self) -> DataLoader:
        """Create DataLoader for evaluation dataset."""
        batch_size = self.eval_params.batch_size
        # Use training num_workers for eval if not provided in EvalParams
        num_workers = getattr(self.eval_params, 'num_workers', self.training_params.num_workers)
        shuffle = True
        eval_dataset = self.prepare_eval_dataset()

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        return eval_dataloader

    def load_model_and_datasets(self) -> None:
        """Load model and datasets in parallel to optimize startup time."""
        logger.info("Initializing model and dataloaders in parallel...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all three tasks
            model_future = executor.submit(self.prepare_model)
            train_dl_future = executor.submit(self.prepare_train_dataloader)
            eval_dl_future = executor.submit(self.prepare_eval_dataloader)
            
            # Wait for all to complete and retrieve results
            try:
                # Add timeout to detect hangs
                self.model, self.checkpoint_data = model_future.result(timeout=300)
                self.train_dataloader = train_dl_future.result(timeout=300)
                self.eval_dataloader = eval_dl_future.result(timeout=300)
                
                logger.info("✅ Model and dataloaders initialized successfully")
            except TimeoutError:
                logger.exception("Timeout during parallel initialization")
                raise
            except Exception as e:
                # Worker threads already log exceptions with traceback, so we just log error here
                logger.error(f"Failed during parallel initialization: {e}")
                raise

    def log_initialization_parameters(self) -> None:
        """Log all initialization parameters for debugging."""
        logger.info(f"Initializing Training controller with parameters:")
        logger.info(f" - Tenant ID: {self.tenant_id}")
        logger.info(f" - WandB Run Name: {self.wandb_run_name}")
        logger.info(f" - Device: {self.device}")
        logger.info(f" - Model Class: {self.model.__class__.__name__}")
        logger.info(f" - Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f" - LR Scheduler: {self.lr_scheduler.__class__.__name__}")
        logger.info(f" - Margin: {self.margin}")
        logger.info(f" - Training Params:")
        for key, value in self.training_params.dict().items():
            logger.info(f"    - {key}: {value}")
        logger.info(f" - Evaluation Params:")
        for key, value in self.eval_params.dict().items():
            logger.info(f"    - {key}: {value}")
            

