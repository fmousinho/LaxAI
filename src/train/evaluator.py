import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, roc_auc_score
from collections import defaultdict
import json

from train.dataset import LacrossePlayerDataset
from config.all_config import wandb_config, evaluator_config
from config.transforms import get_transforms
from train.wandb_logger import wandb_logger

logger = logging.getLogger(__name__)

N_PLAYERS_FOR_VAL = 1
EMBEDDINGS_PER_LOG_MSG = 100

class ModelEvaluator:
    """
    Comprehensive evaluation system for Siamese networks in player re-identification.
    
    Implements multiple evaluation methodologies:
    - Distance-based metrics (Euclidean, Cosine)
    - Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    - Ranking metrics (Rank-1, Rank-5, mAP, CMC curve)
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 threshold: float = evaluator_config.threshold, stop_callback: Optional[Callable] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained Siamese network model
            device: Device to run evaluation on
            threshold: Similarity threshold for binary classification (default 0.5, will be optimized)
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()
        self.stop_callback = stop_callback

    def evaluate_comprehensive(self, dataset) -> Dict[str, Any]:
        """
        Run comprehensive evaluation including all metrics.
        
        Args:
            dataset: LacrossePlayerDataset instance for validation
            use_validation_split: If True, use existing val/ folder; if False, create random split
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")

        if not dataset:
            raise ValueError("dataset is required for evaluation")
        
        logger.info("Creating embeddings")
        embs, labels, image_paths  = self._generate_embeddings(dataset)
        
        sims, dists, labels_eq = self._compute_pairwise_batches(
            embs,
            labels,
            batch_size=4096, 
            compute_distances=True
        )
        
        # Generate embeddings for validation set
        logger.info("Generating embeddings for validation set...")
        embeddings, labels, image_paths = self._generate_embeddings(dataset)
        
        # Distance-based evaluation
        logger.info("Computing distance-based metrics...")
        distance_metrics = self._evaluate_distances(sims, dists, labels_eq)
        
        # Classification evaluation
        logger.info("Computing classification metrics...")
        classification_metrics = self._evaluate_classification(sims, labels_eq)
        
        # Ranking evaluation
        logger.info("Computing ranking metrics...")
        ranking_metrics = self._evaluate_ranking(embeddings, labels)
        
        
        # Aggregate results
        results = {
            'distance_metrics': distance_metrics,
            'classification_metrics': classification_metrics,
            'ranking_metrics': ranking_metrics,
        }
        
        
        # Save detailed results
        self._save_results(results)
        
        logger.info("✅ Comprehensive evaluation completed")
        return results 
    

    def _generate_embeddings(self, dataset: LacrossePlayerDataset, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate embeddings for all images in the dataset in batches.

        Args:
            dataset: LacrossePlayerDataset
            batch_size: batch size for embedding generation

        Returns:
            embeddings: Numpy array of embeddings
            labels: Numpy array of corresponding player labels
            image_paths: List of image identifiers
        """
        import torchvision.transforms as T
        from torch.utils.data import DataLoader

        logger.info(f"Generating embeddings for {len(dataset)} samples in batches of {batch_size}...")

        # Ensure eval mode and no gradients
        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_paths = []

        # Wrap dataset in DataLoader for batching
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=evaluator_config.number_of_workers,       
            pin_memory=True      # improves GPU transfer speed
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Dataset returns could vary depending on your format
                if len(batch) == 4:  
                    images, _, _, labels = batch
                else:
                    images, labels = batch[0], batch[-1]

                # Move to device
                images = images.to(self.device, non_blocking=True)

                # Forward pass
                embeddings = self.model(images)  # already normalized

                # Move to CPU numpy
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.extend([str(l.item()) if torch.is_tensor(l) else str(l) for l in labels])
                all_paths.extend([f"image_{batch_idx * batch_size + i}" for i in range(len(labels))])

                if batch_idx % 10 == 0:
                    logger.info(f"Processed {batch_idx * batch_size}/{len(dataset)} images")

        # Stack results
        embeddings_array = np.vstack(all_embeddings)
        labels_array = np.array(all_labels)

        logger.info(f"Generated {embeddings_array.shape[0]} embeddings for {len(set(labels_array))} unique players")
        logger.info(f"Embeddings shape: {embeddings_array.shape}, mean={embeddings_array.mean():.4f}")

        return embeddings_array, labels_array, all_paths
    

    def _compute_pairwise_batches(
            self,
            embeddings: np.ndarray,
            labels: np.ndarray,
            batch_size: int = 4096,
            compute_distances: bool = True
        ):
        """
        Compute pairwise cosine similarities and (optionally) Euclidean distances in batches on GPU.
        Returns:
            sims: torch.Tensor, shape [N_pairs]
            dists: torch.Tensor or None
            labels_eq: torch.BoolTensor, shape [N_pairs]
        """
        
        device = self.device

        X = torch.tensor(embeddings, dtype=torch.float32, device=device)
        y = torch.tensor(labels, device=device)

        X_norm = torch.nn.functional.normalize(X, dim=1)
        n = X.shape[0]

        sims_list, dists_list, labels_list = [], [], []

        for i in range(0, n, batch_size):
            xi = X[i:i+batch_size]
            yi = y[i:i+batch_size]
            xi_norm = X_norm[i:i+batch_size]

            for j in range(i+1, n, batch_size):
                xj = X[j:j+batch_size]
                yj = y[j:j+batch_size]
                xj_norm = X_norm[j:j+batch_size]

                sims = torch.mm(xi_norm, xj_norm.T)  # cosine similarities
                labels_eq = yi[:, None] == yj[None, :]

                sims_list.append(sims.flatten())
                labels_list.append(labels_eq.flatten())

                if compute_distances:
                    dists = torch.cdist(xi, xj, p=2)
                    dists_list.append(dists.flatten())

        sims = torch.cat(sims_list)
        labels_eq = torch.cat(labels_list)
        dists = torch.cat(dists_list) if compute_distances else None

        return sims, dists, labels_eq


    def _evaluate_distances(self, sims, dists, labels_eq) -> Dict[str, float]:
        """Consume batched engine outputs to compute distance metrics."""
        sims_np = sims.cpu().numpy()
        dists_np = dists.cpu().numpy()
        labels_np = labels_eq.cpu().numpy()

        same_mask = labels_np
        diff_mask = ~labels_np

        avg_dist_same = dists_np[same_mask].mean() if same_mask.any() else float("nan")
        avg_dist_diff = dists_np[diff_mask].mean() if diff_mask.any() else float("nan")
        avg_sim_same = sims_np[same_mask].mean() if same_mask.any() else float("nan")
        avg_sim_diff = sims_np[diff_mask].mean() if diff_mask.any() else float("nan")

        return {
            "avg_distance_same_player": float(avg_dist_same),
            "avg_distance_different_player": float(avg_dist_diff),
            "avg_similarity_same_player": float(avg_sim_same),
            "avg_similarity_different_player": float(avg_sim_diff),
            "distance_separation": float(avg_dist_diff - avg_dist_same),
            "similarity_separation": float(avg_sim_same - avg_sim_diff),
            "same_player_pairs_count": int(same_mask.sum()),
            "different_player_pairs_count": int(diff_mask.sum()),
        }

    def _evaluate_classification(self, sims, labels_eq) -> Dict[str, float]:
        """Classification metrics with optimal threshold."""

        y_scores = sims.cpu().numpy()
        y_true = labels_eq.int().cpu().numpy()

        best_thresh, best_f1 = self._find_optimal_threshold(y_true, y_scores)
        y_pred = (y_scores > best_thresh).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.0

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "threshold_used": float(best_thresh),
        }
    
    def _evaluate_ranking(self, embeddings: np.ndarray, labels: np.ndarray, batch_size: int = 4096) -> Dict[str, float]:
        """
        Compute mean Average Precision (mAP) for retrieval.
        Uses batched cosine similarities on GPU.
        """
        device = self.device
        X = torch.tensor(embeddings, dtype=torch.float32, device=device)
        y = torch.tensor(labels, device=device)
        X_norm = torch.nn.functional.normalize(X, dim=1)

        n = X.shape[0]
        ap_list = []

        for i in range(n):
            anchor = X_norm[i:i+1]  # [1, d]
            sims = torch.mm(anchor, X_norm.T).squeeze(0)  # [n]
            sims[i] = -1e9  # mask self

            # Sort by similarity
            sorted_idx = torch.argsort(sims, descending=True)
            sorted_labels = y[sorted_idx].cpu().numpy()

            true_labels = y[i].cpu().item()
            relevant = (sorted_labels == true_labels).astype(int)

            ap = self._compute_average_precision(relevant)
            ap_list.append(ap)

        mAP = float(np.mean(ap_list)) if ap_list else 0.0
        return {"mean_average_precision": mAP}


    def _compute_average_precision(self, relevant: np.ndarray) -> float:
        """Compute AP for a ranked list of binary relevance (1 if correct, 0 if not)."""
        hits = np.where(relevant == 1)[0]
        if len(hits) == 0:
            return 0.0

        precisions = [(i+1) / (rank+1) for i, rank in enumerate(hits)]
        return float(np.mean(precisions))


    

    def _find_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
        best_f1 = 0.0
        best_threshold = self.threshold
        
        # Pre-calculate total positives in ground truth
        total_positives = np.sum(y_true)
        
        for threshold in thresholds:
            y_pred_temp = (y_scores > threshold)
            
            tp = np.sum((y_pred_temp == 1) & (y_true == 1))
            
            # Calculate precision and recall manually
            precision = tp / np.sum(y_pred_temp) if np.sum(y_pred_temp) > 0 else 0
            recall = tp / total_positives if total_positives > 0 else 0
            
            f1_temp = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if f1_temp > best_f1:
                best_f1 = f1_temp
                best_threshold = float(threshold)
                
        return best_threshold, best_f1
    
    
    
    def _log_to_wandb(self, results: Dict[str, Any]) -> None:
        """
        Log evaluation results to Weights & Biases.
        """
        if not wandb_config.enabled:
            return
        
        # Flatten the results for wandb logging
        wandb_metrics = {}
        
        # Distance metrics
        for key, value in results['distance_metrics'].items():
            wandb_metrics[f'eval/distance/{key}'] = value
        
        # Classification metrics
        for key, value in results['classification_metrics'].items():
            wandb_metrics[f'eval/classification/{key}'] = value
        
        # Ranking metrics
        for key, value in results['ranking_metrics'].items():
            wandb_metrics[f'eval/ranking/{key}'] = value
        
        wandb_logger.log_metrics(wandb_metrics)
        logger.info("Evaluation results logged to wandb")
    
    def _save_results(self, results: Dict[str, Any], save_dir: str = "evaluation_results") -> None:
        """
        Save detailed evaluation results to disk.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Convert all numpy types to native Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        results_converted = convert(results)

        # Save JSON results
        results_file = os.path.join(save_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_converted, f, indent=2)

        logger.info(f"Evaluation results saved to: {results_file}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        
        # Classification metrics
        cls_metrics = results['classification_metrics']
        report.append(f"\nClassification Metrics:")
        report.append(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
        report.append(f"  Precision: {cls_metrics['precision']:.4f}")
        report.append(f"  Recall: {cls_metrics['recall']:.4f}")
        report.append(f"  F1-Score: {cls_metrics['f1_score']:.4f}")
        report.append(f"  ROC-AUC: {cls_metrics['roc_auc']:.4f}")
        
        # Ranking metrics
        rank_metrics = results['ranking_metrics']
        report.append(f"\nRanking Metrics:")
        report.append(f"  Rank-1 Accuracy: {rank_metrics['rank_1_accuracy']:.4f}")
        report.append(f"  Rank-5 Accuracy: {rank_metrics['rank_5_accuracy']:.4f}")
        report.append(f"  Mean Average Precision: {rank_metrics['mean_average_precision']:.4f}")
        
        # Distance metrics
        dist_metrics = results['distance_metrics']
        report.append(f"\nDistance Analysis:")
        report.append(f"  Avg Dist (Same Player): {dist_metrics['avg_distance_same_player']:.4f}")
        report.append(f"  Avg Dist (Different Player): {dist_metrics['avg_distance_different_player']:.4f}")
        report.append(f"  Avg Sim (Same Player): {dist_metrics['avg_similarity_same_player']:.4f}")
        report.append(f"  Avg Sim (Different Player): {dist_metrics['avg_similarity_different_player']:.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
