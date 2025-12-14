"""
Metrics computation and logging for training.

Pairs with schemas.metrics.MetricsData for structured metric storage.

IMPORTANT: When updating new metrics, make sure to update the __add__ and __truediv__ 
methods in MetricsData accordingly, and add logging in _log_to_logger and _log_to_wandb.


"""


import logging
logger = logging.getLogger(__name__)

from typing import Dict, Optional, Tuple
import torch

from schemas.metrics import MetricsData, EvalData


class Metrics:

    def __init__(self, wandb_logger = None):
    
        self.batch_metrics = MetricsData()
        self.epoch_metrics = MetricsData()
        self.epoch_accumulations = MetricsData()
        self.running_num_batches_in_epoch = 0

        self.eval_batch_metrics = EvalData()
        self.eval_epoch_metrics = EvalData()
        self.eval_epoch_accumulations: Optional[EvalData] = None
        self.running_num_eval_batches_in_epoch = 0

        self.eval_metrics_available = False
       
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
        """Accumulate batch metrics into epoch metrics."""

        with torch.no_grad():
            self.batch_metrics.loss = batch_loss
            self.batch_metrics.gradient_norm = self._compute_gradient_norm(model)

            positive_distance = torch.norm(anchor_embs - positive_embs, p=2, dim=1)
            negative_distance = torch.norm(anchor_embs - negative_embs, p=2, dim=1)
            
            self.batch_metrics.positive_distance = positive_distance.mean().item()
            self.batch_metrics.negative_distance = negative_distance.mean().item()
            (self.batch_metrics.mining_hard, 
                self.batch_metrics.mining_semi_hard, 
                self.batch_metrics.mining_easy) = self._compute_mining_efficiency(
                    positive_distance, negative_distance, margin
                )
            self.batch_metrics.model_variance = self._compute_embeddings_variance(anchor_embs)
            self._accumulate_epoch_metrics(epoch, self.batch_metrics)

    def finalize_epoch_metrics(self, epoch: int, learning_rate: Optional[float] = None, weight_decay: Optional[float] = None):
        """
        Calculates epoch metrics based on acculated averages.
        Logs metrics to standard logger and Weights & Biases if provided.
        
        Args:
            epoch: Current epoch number
            learning_rate: Current learning rate value
            weight_decay: Weight decay value
        """
        if self.running_num_batches_in_epoch == 0:
            logger.warning(f"No batches processed in epoch {epoch}. Cannot finalize metrics.")
            return

        # Use operator overloading for cleaner averaging
        self.epoch_metrics = self.epoch_accumulations / self.running_num_batches_in_epoch
        
        # Set learning rate and weight decay directly in epoch_metrics
        if learning_rate is not None:
            self.epoch_metrics.learning_rate = learning_rate
        if weight_decay is not None:
            self.epoch_metrics.weight_decay = weight_decay

        self._maybe_log_to_wandb(epoch)
        self._log_to_logger(epoch)
        

        # Reset accumulations for next epoch
        self.epoch_accumulations = MetricsData()
        self.running_num_batches_in_epoch = 0

        # Eval accuulations reset needs cannot be done in finalize_eval_epoch_metrics
        # because it is must be called after common _maybe_log_to_wandb
        if self.eval_metrics_available:
            self._log_eval_to_logger(epoch)
            self.eval_epoch_accumulations = EvalData()
            self.running_num_eval_batches_in_epoch = 0
            self.eval_metrics_available = False

    def update_eval_batch_data(
            self, 
            epoch: int, 
            labels: torch.Tensor,
            embeddings: torch.Tensor
        ):
        """Accumulate evaluation batch metrics into epoch metrics."""
        with torch.no_grad():
            
            centroids = self._compute_centroids(labels, embeddings)
            confusion = self._compute_confusion_matrix(labels, embeddings, centroids)

            # Calculate Macro-Averaged Metrics (Per-Class)
            tp = torch.diagonal(confusion).float()
            col_sums = confusion.sum(dim=0).float()
            row_sums = confusion.sum(dim=1).float()
            
            fp = col_sums - tp
            fn = row_sums - tp
            
            # Precision per class: TP / (TP + FP)
            # Avoid division by zero
            precision_per_class = torch.zeros_like(tp)
            denom_p = tp + fp
            mask_p = denom_p > 0
            precision_per_class[mask_p] = tp[mask_p] / denom_p[mask_p]
            
            # Recall per class: TP / (TP + FN)
            recall_per_class = torch.zeros_like(tp)
            denom_r = tp + fn
            mask_r = denom_r > 0
            recall_per_class[mask_r] = tp[mask_r] / denom_r[mask_r]
            
            # F1 per class: 2 * P * R / (P + R)
            f1_per_class = torch.zeros_like(tp)
            denom_f1 = precision_per_class + recall_per_class
            mask_f1 = denom_f1 > 0
            f1_per_class[mask_f1] = 2 * (precision_per_class[mask_f1] * recall_per_class[mask_f1]) / denom_f1[mask_f1]
            
            # Store Macro-Averages
            self.eval_batch_metrics.precision = precision_per_class.mean().item()
            self.eval_batch_metrics.recall = recall_per_class.mean().item()
            self.eval_batch_metrics.f1_score = f1_per_class.mean().item()
            
            self.eval_batch_metrics.k1 = self._compute_k1(labels, embeddings, centroids)
            self.eval_batch_metrics.k5 = self._compute_k5(labels, embeddings, centroids)
            self.eval_batch_metrics.mean_avg_precision = self._compute_map(labels, embeddings, centroids)

            self._accumulate_eval_epoch_metrics(epoch, self.eval_batch_metrics)


    def finalize_eval_epoch_metrics(self, epoch: int):
        """Calculates evaluation epoch metrics based on accumulated averages.
        Logging is done by finalize_epoch_metrics.
        """
        if self.running_num_eval_batches_in_epoch == 0:
            logger.warning(f"No evaluation batches processed in epoch {epoch}. Cannot finalize eval metrics.")
            return

        # Use operator overloading for cleaner averagingse
        if self.eval_epoch_accumulations is not None:
            self.eval_epoch_metrics = self.eval_epoch_accumulations / self.running_num_eval_batches_in_epoch
            self.eval_metrics_available = True
   


    # ===== Computation Helpers =====

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
        ) -> Tuple[float, float, float]:
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

    def _accumulate_eval_epoch_metrics(self, epoch: int, eval_batch_metrics: EvalData):
        """Accumulate evaluation batch metrics into evaluation epoch metrics."""
        if self.eval_epoch_accumulations is None:
            self.eval_epoch_accumulations = EvalData()
        self.eval_epoch_accumulations += eval_batch_metrics
        self.running_num_eval_batches_in_epoch += 1

    def _compute_centroids(self, labels: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute centroids: average embeddings for each unique label."""
        unique_labels = torch.unique(labels)
        centroids = []
        
        for label in unique_labels:
            mask = labels == label
            label_embeddings = embeddings[mask]
            centroid = label_embeddings.mean(dim=0)
            centroids.append(centroid)
        
        return torch.stack(centroids)  # Shape: [num_classes, embedding_dim]

    def _compute_confusion_matrix(
            self,
            labels: torch.Tensor, 
            embeddings: torch.Tensor, 
            centroids: torch.Tensor
        ) -> torch.Tensor:
        """Compute confusion matrix based on nearest centroid classification."""
        num_classes = centroids.size(0)
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32).to(embeddings.device)
        
        # Map unique labels to 0..N-1 indices
        unique_labels = torch.unique(labels)
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        
        for i in range(embeddings.size(0)):
            embedding = embeddings[i]
            label = labels[i]
            # Compute distances to centroids
            dists = torch.norm(centroids - embedding.unsqueeze(0), p=2, dim=1)
            predicted_label_idx = int(torch.argmin(dists).item())
            
            true_label_idx = label_to_idx[label.item()]
            confusion_matrix[true_label_idx, predicted_label_idx] += 1
        return confusion_matrix


    
    def _compute_k1(
            self, 
            labels: torch.Tensor, 
            embeddings: torch.Tensor, 
            centroids: torch.Tensor
        ) -> float:
        """Compute K@1 (top-1 accuracy) retrieval metric using centroids.
        
        For each embedding, check if the nearest centroid matches its true label.
        """
        # Compute distance matrix: [num_samples, num_centroids]
        # Using cdist is more efficient than nested loops
        distances = torch.cdist(embeddings, centroids, p=2)
        
        # Get predicted labels (nearest centroid)
        predicted = torch.argmin(distances, dim=1)
        
        # Get unique label mapping for indexing
        unique_labels = torch.unique(labels)
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        true_indices = torch.tensor([label_to_idx[l.item()] for l in labels], device=labels.device)
        
        # Compute accuracy
        correct = (predicted == true_indices).float()
        return correct.mean().item()
    
    def _compute_k5(
            self, 
            labels: torch.Tensor, 
            embeddings: torch.Tensor, 
            centroids: torch.Tensor
        ) -> float:
        """Compute K@5 retrieval metric using centroids.
        
        For each embedding, check if true label's centroid is in top-5 closest.
        """
        # Compute distance matrix: [num_samples, num_centroids]
        distances = torch.cdist(embeddings, centroids, p=2)
        
        # Get top-5 nearest centroids for each embedding
        _, top5_indices = torch.topk(distances, k=min(5, centroids.size(0)), dim=1, largest=False)
        
        # Get unique label mapping
        unique_labels = torch.unique(labels)
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        true_indices = torch.tensor([label_to_idx[l.item()] for l in labels], device=labels.device)
        
        # Check if true label is in top-5
        true_in_top5 = (top5_indices == true_indices.unsqueeze(1)).any(dim=1).float()
        return true_in_top5.mean().item()
    
    def _compute_map(
            self, 
            labels: torch.Tensor, 
            embeddings: torch.Tensor, 
            centroids: torch.Tensor
        ) -> float:
        """Compute Mean Average Precision (mAP) using centroids.
        
        For each embedding, compute AP based on ranking of all centroids.
        """
        # Compute distance matrix and sort: [num_samples, num_centroids]
        distances = torch.cdist(embeddings, centroids, p=2)
        sorted_indices = torch.argsort(distances, dim=1)
        
        # Get unique label mapping
        unique_labels = torch.unique(labels)
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        true_indices = torch.tensor([label_to_idx[l.item()] for l in labels], device=labels.device)
        
        # Compute AP for each sample
        aps = []
        for i in range(embeddings.size(0)):
            sorted_preds = sorted_indices[i]
            true_idx = true_indices[i]
            
            # Find position of correct label (1-indexed for AP calculation)
            position = (sorted_preds == true_idx).nonzero(as_tuple=True)[0].item() + 1
            
            # AP = 1 / position (simplified for single relevant item)
            ap = 1.0 / position
            aps.append(ap)
        
        return sum(aps) / len(aps)




    # ===== Logging Helpers =====

    def _log_to_logger(self, epoch: int):
        """Log epoch metrics to standard logger."""
        logger.info(f"🧮 Epoch {epoch} Metrics - ")
        for field_name, value in self.epoch_metrics.model_dump().items():
            logger.info(f"  - {field_name}: {value:.6f}")

    def _log_eval_to_logger(self, epoch: int):
        """Log evaluation epoch metrics to standard logger."""
        logger.info(f"🧮 Eval Epoch {epoch} Metrics - ")
        for field_name, value in self.eval_epoch_metrics.model_dump().items():
            logger.info(f"  - {field_name}: {value:.6f}")
            

    def _maybe_log_to_wandb(self, epoch: int):
        """Log epoch metrics to Weights & Biases if logger is provided."""
        if self.wandb_logger is not None:
            
            metrics_dict = {}
            for field_name, value in self.epoch_metrics.model_dump().items():
                metrics_dict[f"train/{field_name}"] = value

            # Eval metrics are not updated in every epoch
            if self.eval_metrics_available:
                eval_dict = {}
                for field_name, value in self.eval_epoch_metrics.model_dump().items():
                    eval_dict[f"eval/{field_name}"] = value
                metrics_dict.update(eval_dict)
            
            self.wandb_logger.log_metrics(metrics_dict, step=epoch)
