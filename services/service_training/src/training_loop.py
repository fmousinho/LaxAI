import logging
logger = logging.getLogger(__name__)

import os
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shared_libs.utils.gpu_memory import clear_gpu_memory
from metrics import Metrics



# Constants
BATCHES_PER_LOG_MSG = 10
EPOCHS_PER_VAL = 50
EPOCHS_PER_CHECKPOINT = 10


class TrainingLoop:
    """Triplet-loss trainer with staged LR scheduling and aggressive memory hygiene."""
    
    def __init__(self, 
                model: nn.Module,
                train_dataloader: DataLoader,
                eval_dataloader: Optional[DataLoader],
                loss_fn: Callable,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: Any,
                starting_epoch: int = 1,
                num_epochs: int = 10,
                wandb_logger = None,
                device: Optional[torch.device] = None,
                task_id: Optional[str] = None
                ):
        """
        Initialize the training class with hyperparameters and runtime knobs.
        
        Args:
            device: Torch device to run the model on (CPU, GPU, or MPS)
            training_parameters: TrainingParams object containing all training hyperparameters
        
        All hyperparameters default to values from training_config/wandb_config 
        but can be overridden via kwargs.
        """

        # Clear GPU memory at start to recover from previous crashes
        clear_gpu_memory()
        logger.info("Cleared GPU memory on Training initialization")

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training device: {self.device}")

        self.model = model
        self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer.to
        self.lr_scheduler = lr_scheduler
        self.current_epoch = starting_epoch
        self.num_epochs = num_epochs
        self.wandb_logger = wandb_logger
        self.metrics = Metrics(wandb_logger=wandb_logger)
        self.task_id = task_id


    def train(self, task_id: str, cancellation_requested_fn: Callable[[], bool]):
        """
        Main training loop for triplet-loss model.
        
        Args:
            task_id: Unique identifier for the training task
            cancellation_requested_fn: A callable that returns True if training should be cancelled
            
        Returns:
            The trained model
            
        Raises:
            RuntimeError: If required components are not setup
          
        """
        # ========================================================================
        # Sanity checks
        # ========================================================================
        if self.model is None or self.train_dataloader is None:
            raise RuntimeError("Model and dataloader must be setup before training")
        
        if self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Optimizer and loss function must be setup before training")
        
        self.cancellation_requested = cancellation_requested_fn
        
        # ========================================================================
        # Training setup and logging
        # ========================================================================
 
        ttl_batches = len(self.train_dataloader)

        starting_epoch = self.current_epoch

        try: 
            for epoch in range(starting_epoch, self.num_epochs + 1):
                logger.info(f"=== Epoch {epoch}/{self.num_epochs} ===")
                
                if not self.model.training:
                    self.model.train() 

                self.margin = self.update_margin (epoch, self.margin)

                for i, (anchor, positive, negative, labels) in enumerate(self.train_dataloader):

                    if (i + 1) % BATCHES_PER_LOG_MSG == 0:
                        logger.info(f"Training Batch {i+1}/{ttl_batches}")

                    self.optimizer.zero_grad()

                    anchor = anchor.to(self.device, non_blocking=True)
                    positive = positive.to(self.device, non_blocking=True)
                    negative = negative.to(self.device, non_blocking=True)

                    anchor_embs = self.model.forward(anchor) 
                    positive_embs = self.model.forward(positive)  
                    negative_embs = self.model.forward(negative) 

                    batch_loss = self.loss_fn(anchor_embs, positive_embs, negative_embs, self.margin)
                    batch_loss.backward()

                    self.metrics.update_with_batch_data(epoch, anchor_embs, positive_embs, negative_embs, batch_loss, self.margin)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                self.maybe_save_checkpoint(epoch, self.model, self.optimizer, self.lr_scheduler)

                if self.cancellation_requested():
                    logger.info(f"--- Cancellation requested at Epoch {epoch}. Stopping training. ---")
                    self.save_checkpoint()
                    return

                self.maybe_run_mid_train_evaluation(epoch, self.val_dataloader)
                
                if self.cancellation_requested():
                    logger.info(f"--- Cancellation requested at Epoch {epoch}. Stopping training. ---")
                    self.save_checkpoint()
                    return

                self.metrics.finalize_epoch_metrics(epoch)

                self.current_epoch += 1

            logger.info("🎉 Training Complete!!!")
            return self.model

        except Exception as e:
            error_msg = f"Training loop failed at epoch {epoch}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
                
    def maybe_run_mid_train_evaluation(self):
        if self.current_epoch % EPOCHS_PER_VAL == 0:
            logger.info(f"--- Running Validation after Epoch {self.current_epoch} ---")
            self.mid_train_evaluation()

    def mid_train_evaluation(self):
        """Run evaluation at mid-training. """
        if self.model.training:
            self.model.eval()  
        with torch.no_grad():
           for i, (player, crop) in enumerate(self.eval_dataloader):
                crop = crop.to(self.device, non_blocking=True)
                emb = self.model.forward(crop)
                self.metrics.update_eval_batch_data(
                    self.current_epoch, player, emb)
                if self.cancellation_requested():
                    return
            self.metrics.finalize_eval_epoch_metrics(self.current_epoch)

    def update_margin(self, epoch: int, current_margin: float) -> float:
        """Update the triplet loss margin based on epoch schedule."""
        return current_margin

    def maybe_save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: Any):
        """Save model checkpoint at regular intervals."""
        if epoch % EPOCHS_PER_CHECKPOINT == 0:
            self.save_checkpoint()

    def save_checkpoint(self):      
            logger.info(f"--- Saving Checkpoint at Epoch {self.current_epoch} ---")
            if not self.wandb_logger:
                logger.warning("WandB logger not initialized, skipping checkpoint save.")
                return
            state_dicts = {
                'model_state_dict': self.model.state_dict(),           
                'optimizer_state_dict': self.optimizer.state_dict(),   
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),    
            }
            self.wandb_logger.save_checkpoint(
                state_dicts=state_dicts,
                epoch=self.current_epoch,
                task_id=self.task_id
            )


        
    