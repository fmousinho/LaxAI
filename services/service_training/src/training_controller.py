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
        self.task_id = task_id

        self.storage_client = get_storage(tenant_id)
        self.path_manager = GCSPaths()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.training_params = training_params
        self.eval_params = eval_params

        self.wandb_logger = wandb_logger
        # Initialize WandB with required run name
        self.setup_wandb_logger(run_name=self.wandb_run_name)

        self.starting_epoch = 1
        self.loss_fn = loss_fn

        # Initialize to None before loading (will be set by load_model_and_datasets)
        self.model = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.checkpoint_data = None
        
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

        self.training_loop = TrainingLoop(
            model=self.model,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            loss_fn=self.loss_fn,
            optimizer = self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=self.device,
            starting_epoch=self.starting_epoch,
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
        """Setup WandB logger for the training pipeline."""
        config = {
            "tenant_id": self.tenant_id,
            "model_class": self.model.__class__.__name__,
            "loss_function": self.loss_fn.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "lr_scheduler": self.lr_scheduler.__class__.__name__,
            "training_params": {**self.training_params.dict()},
            "eval_params": {**self.eval_params.dict()}
        }
        self.wandb_logger.init_run(config=config, run_name=run_name)
    
    def prepare_model(self) -> Tuple[torch.nn.Module, Optional[Dict[str, Any]]]:
        """Setup the model class and training/evaluation parameters."""
        weights_source = self.training_params.weights
        model = None
        checkpoint_data = None

        try: 
            model_name = ReIdModel.model_name
            
            if weights_source == "checkpoint":
                logger.info("Attempting to load model weights from checkpoint")
                # load_checkpoint returns Optional[StateDicts]
                # We pass None to let it auto-detect the checkpoint name based on the current run
                checkpoint_data = self.wandb_logger.load_checkpoint()
                
                if checkpoint_data:
                    model = ReIdModel(pretrained=False)
                    # Load model state
                    model.load_state_dict(checkpoint_data['model_state_dict'])
                    
                    # Set starting epoch
                    self.starting_epoch = checkpoint_data.get('epoch', 0) + 1
                    logger.info(f"Resuming training from epoch {self.starting_epoch}")
                else:
                    logger.warning(f"No checkpoint found for run {self.wandb_run_name}, falling back to 'latest'")
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
            

    def prepare_train_dataset(self) -> Dataset:
        """Prepare training dataset with triple (anchor, positive, negative) sampling."""
        try:
            train_transform = get_transforms('training')
            dataset_addresses = self.training_params.dataset_address
            if type(dataset_addresses) is str:
                dataset_addresses = [dataset_addresses]
            logger.info("Training dataset built with the following addresses:")
            adjusted_addresses = [addr.rstrip("/") + "/train/" for addr in dataset_addresses]
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
            eval_transform = get_transforms('evaluation')
            dataset_addresses = self.training_params.dataset_address
            if type(dataset_addresses) is str:
                dataset_addresses = [dataset_addresses]
            largest_dataset_path: str = ""
            largest_size: int = 0
            largest_dataset_images_paths = set()
            for dataset_address in dataset_addresses:
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
        logger.info(f" - Training Params:")
        for key, value in self.training_params.dict().items():
            logger.info(f"    - {key}: {value}")
        logger.info(f" - Evaluation Params:")
        for key, value in self.eval_params.dict().items():
            logger.info(f"    - {key}: {value}")
            

