import logging
logger = logging.getLogger(__name__)

import os
import random
import traceback
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
from shared_libs.utils.id_generator import generate_simple_uuid


from schemas.training import TrainingParams, EvalParams
from wandb_logger import wandb_logger
from training_loop import TrainingLoop
from loss_fn import loss_fn
from model import ReIdModel
from eval_dataset.py import load_eval_dataset
from dataset import LacrossePlayerDataset



class TrainingController():

    def __init__(
            self, 
            tenant_id: str, 
            wandb_run_name: str,
            training_params: TrainingParams = TrainingParams(),
            eval_params: EvalParams = EvalParams()
        ):
        """
        Sets up all components required for training.
        """

        self.log_initialization_parameters()

        self.storage_client = get_storage(tenant_id)
        self.path_manager = GCSPaths()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.training_params = training_params
        self.eval_params = eval_params

        self.wandb_run_name = wandb_run_name
        self.wandb_logger = wandb_logger
        self.setup_wandb_logger()

        self.starting_epoch = 1
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.AdamW (lr=self.training_params.lr_initial)
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=self.training_params.lr_scheduler_factor,
            patience=self.training_params.lr_scheduler_patience,
        )

        self.model: torch.nn.Module
        self.train_dataloader: DataLoader
        self.eval_dataloader: DataLoader    
        self.load_model_and_datasets()

        self.training_loop = TrainingLoop(
            model=self.model,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            loss_fn=self.loss_fn,
            optimizer = self.optimizer,
            lr_scheduler=self.lr_scheduler,
            device=self.device,
            starting_epoch=self.starting_epoch,
            training_params=self.training_params,
            wandb_logger=self.wandb_logger
        )

        self.training_threads: Dict[str, threading.Thread] = {}
        self.cancellation_requested_flag = False

    def train(self) -> str:
        """Start the training process in a background thread.

        Returns:
            task_id: Simple UUID identifying this training run.
        """
        task_id = generate_simple_uuid()
        logger.info(f"Starting training task {task_id} (threaded) with WandB run '{self.wandb_run_name}'")

        def _cancellation_requested() -> bool:
            return self.cancellation_requested_flag

        def _run_training():
            try:
                self.training_loop.train(task_id, cancellation_requested_fn=_cancellation_requested)
                self.wandb_logger.finish()
            except Exception as e:
                logger.error(f"Training task {task_id} failed: {e}")
                traceback.print_exc()

        # Spawn daemon thread so it won't block process shutdown
        self.training_threads[task_id] = threading.Thread(target=_run_training, name=f"train-{task_id}", daemon=True)
        self.training_threads[task_id].start()
        return task_id

    def graceful_cancellation_request(self, task_id: str, timout: int = 600) -> bool:
        """Waits for graceful cancellation until timeout is reached."""
        logger.info(f"Graceful cancellation requested for training task {task_id}")
        self.cancellation_requested_flag = True

        def _cancellation_poller():
            check_interval = 5  # seconds
            wait_time = 0
            while self.training_thread[task_id].is_alive():
                if wait_time >= timout:
                    logger.error(f"CANCELLATION TIMOUT REACHED and {task_id} is still alive. Consider killing the process externally.")
                time.sleep(check_interval)
                wait_time += check_interval

        threading.Thread(target=_cancellation_poller, name=f"cancel-poller-{task_id}", daemon=True).start()
        return True

    def train_is_running(self, task_id: str) -> bool:
        """Check if the training task is still running."""
        thread = self.training_threads.get(task_id)
        if thread:
            return thread.is_alive()
        return False


    def setup_wandb_logger(self, run_name):
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
        self.wandb_logger.init_run(
            name=self.wandb_run_name,
            config=config
            )
    
    def prepare_model(self) -> nn.Module:
        """Setup the model class and training/evaluation parameters."""
        weights_source = self.training_params.weights

        try: 
            model_name = ReIdModel.model_name
            
            if weights_source == "checkpoint":
                logger.info("Attempting to load model weights from checkpoint")
                model, last_epoch = self.wandb_logger.load_from_checkpoint(self.wandb_run_name)
                if not self.model:
                    logger.warning(f"No checkpoint found in run {self.wandb_run_name}")
                    weights_source = "latest"
                self.starting_epoch = last_epoch + 1
            
            if weights_source == "latest":
                logger.info("Attempting to load weights from latest saved model")
                model = self.wandb_logger().load_model(model_name=model_name, alias="latest")
                if not self.model:
                    logger.warning("No latest saved model found")
                    weights_source = "reset"
            
            if weights_source == "reset":
                logger.warning("⚠️  Resetting model weights to ResNet defaults as per training parameters")
                model = ReIdModel()

            model.to(self.device)
            return model

        except Exception as e:
            msg = f"Error setting up model with weights source '{weights_source}': {e}"
            logger.error(msg)
            traceback.print_exc()
            raise RuntimeError(msg) from e
            

    def prepare_train_dataset(self) -> Dataset:
        """Prepare training dataset with triple (anchor, positive, negative) sampling."""
        try:
            train_transform, _ = get_transforms('training')
            dataset_addresses = self.training_params.dataset_address
            if type(dataset_addresses) is str:
                dataset_addresses = [dataset_addresses]
            logger.info("Training dataset built with the following addresses:")
            for i in range(len(dataset_addresses)):
                dataset_addresses[i] = dataset_addresses[i].rstrip("/") + "/train/"
                logger.info(f" - {dataset_addresses[i]}")

            train_dataset = LacrossePlayerDataset(
                storage_client=self.storage_client,
                image_dir=dataset_addresses,
                transform=train_transform,
            )
            return train_dataset

    def prepare_eval_dataset(self) -> Dataset:
        """
        Prepare evaluation dataset. Differently from the training dataset,
        there is no triple sampling. Only the largest val dataset is used in order
        to prevent player overlap between two different val sets (each player is 
        expected to be unique in the eval dataset).
        """
        try:
            _, eval_transform = get_transforms('evaluation')
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
            logger.info(f"Evaluation dataset built from {largest_dataset_path} with {largest_size} images")
            eval_dataset = load_eval_dataset(
                storage_client=self.storage_client,
                image_paths=largest_dataset_path,
                transform=eval_transform
            )
            return eval_dataset
        except Exception as e:
            msg = f"Error preparing evaluation dataset: {e}"
            logger.error(msg)
            traceback.print_exc()
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
            num_workers = self.eval_params.num_workers
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
                    self.model = model_future.result()
                    self.train_dataloader = train_dl_future.result()
                    self.eval_dataloader = eval_dl_future.result()
                    logger.info("✅ Model and dataloaders initialized successfully")
                except Exception as e:
                    logger.error(f"Failed during parallel initialization: {e}")
                    raise

        def log_initialization_parameters(self) -> None:
            """Log all initialization parameters for debugging."""
            logger.info(f"Initializing Training pipeline with parameters:")
            logger.info(f" - Tenant ID: {self.tenant_id}")
            logger.info(f" - WandB Run Name: {self.wandb_run_name}")
            logger.info(f" - Device: {self.device}")
            logger.info(f" - Model Class: {self.model_class.__name__}")
            logger.info(f" - Optimizer: {self.optimizer.__class__.__name__}")
            logger.info(f" - LR Scheduler: {self.lr_scheduler.__class__.__name__}")
            logger.info(f" - Training Params:")
            for key, value in self.training_params.dict().items():
                logger.info(f"    - {key}: {value}")
            logger.info(f" - Evaluation Params:")
            for key, value in self.eval_params.dict().items():
                logger.info(f"    - {key}: {value}")
            

