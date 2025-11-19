"""
Metrics computation and logging for training.

Pairs with schemas.metrics.MetricsData for structured metric storage.

IMPORTANT: When updating new metrics, make sure to update the __add__ and __truediv__ 
methods in MetricsData accordingly, and add logging in _log_to_logger and _log_to_wandb.


"""




import logging
logger = logging.getLogger(__name__)

from typing import Dict
from schemas.metrics import MetricsData

import torch

class Metrics:

    def __init__(self, wandb_logger: Optional = None):
    
        self.batch_metrics = MetricsData()
        self.epoch_metrics = MetricsData()
        self.epoch_accumulations = MetricsData()
        self.running_num_batches_in_epoch = 0
        self.wandb_logger = wandb_logger

    def update_with_batch_data (
            self, model: torch.nn.Module, 
            epoch: int, 
            anchor_embs: torch.Tensor, 
            positive_embs: torch.Tensor, 
            negative_embs: torch.Tensor, 
            batch_loss: float, 
            margin: float
         ):
        """Update metrics with batch data."""

        with torch.no_grad():
            self.batch_metrics.loss = batch_loss.item()
            self.batch_metrics.gradient_norm = self._compute_gradient_norm(model)

            positive_distance = torch.norm(anchor_embs - positive_embs, p=2, dim=1)
            negative_distance = torch.norm(anchor_embs - negative_embs, p=2, dim=1)
            
            self.batch_metrics.positive_distance = positive_distance.mean().item()
            self.batch_metrics.negative_distance = negative_distance.mean().item()
            (self.batch_metrics.mining_hard, 
                self.batch_metrics.mining_semihard, 
                self.batch_metrics.mining_easy) = self._compute_mining_efficiency(
                    positive_distance, negative_distance, margin
                )
            self.batch_metrics.model_variance = self._compute_embeddings_variance(anchor_embs)
            self._accumulate_epoch_metrics(epoch, self.batch_metrics)

    def finalize_epoch_metrics(self, epoch: int):
        """Finalize and log epoch metrics."""
        if self.running_num_batches_in_epoch == 0:
            logger.warning(f"No batches processed in epoch {epoch}. Cannot finalize metrics.")
            return

        # Use operator overloading for cleaner averaging
        self.epoch_metrics = self.epoch_accumulations / self.running_num_batches_in_epoch

        self._maybe_log_to_wandb(epoch, self.epoch_metrics)
        self._log_to_logger(epoch, self.epoch_metrics)

        # Reset accumulations for next epoch
        self.epoch_accumulations = MetricsData()
        self.running_num_batches_in_epoch = 0


    def _compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """Compute the gradient norm of the model."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _compute_mining_efficiency(
            self, 
            pos_dist: torch.Tensor, 
            neg_dist: torch.Tensor, 
            margin: float
        ) -> Dict[str, float]:
        """Compute mining efficiency for triplet loss.
        
        Since pos_dist and neg_dist are already averaged scalars,
        we return simple ratios based on margin thresholds.
        """
        # Calculate distance difference
        hard_count = ((neg_dist - pos_dist) < margin).sum().item()
        semi_hard_count = (((neg_dist - pos_dist) >= margin) & ((neg_dist - pos_dist) < margin + 0.2)).sum().item()
        easy_count = (((neg_dist - pos_dist) >= margin + 0.2)).sum().item()
        total_count = pos_dist.size(0)
        if total_count == 0:
            return (0.0, 0.0, 0.0)
        else:
            return (
                hard_count / total_count,
                semi_hard_count / total_count,
                easy_count / total_count
            )

    def _compute_embeddings_variance(self, embeddings: torch.Tensor) -> float:
        """Compute variance of model parameters."""
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        return torch.var(embeddings, dim=0).mean().item()
    
    def _accumulate_epoch_metrics(self, epoch: int, batch_metrics: MetricsData):
        """Accumulate batch metrics into epoch metrics."""
        self.epoch_accumulations += batch_metrics
        self.running_num_batches_in_epoch += 1


    def _log_to_logger(self, epoch: int, epoch_metrics: MetricsData):
        """Log epoch metrics to standard logger."""
        logger.info(f"🧮 Epoch {epoch} Metrics - ")
        for field_name, value in epoch_metrics.model_dump().items():
            logger.info(f". - {field_name}: {value}")
            

    def _maybe_log_to_wandb(self, epoch: int, epoch_metrics: MetricsData, phase: str = "train"):
        """Log epoch metrics to Weights & Biases if logger is provided."""
        if self.wandb_logger is not None:
            metrics_dict = {}
            for field_name, value in epoch_metrics.model_dump().items():
                metrics_dict[f"{phase}/{field_name}"] = value
            self.wandb_logger.log_metrics(metrics_dict, step=epoch)

