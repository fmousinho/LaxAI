import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
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
                 threshold: float = evaluator_config.threshold):
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
        
        
        # Generate embeddings for validation set
        logger.info("Generating embeddings for validation set...")
        embeddings, labels, image_paths = self._generate_embeddings(dataset)
        
        # Distance-based evaluation
        logger.info("Computing distance-based metrics...")
        distance_metrics = self._evaluate_distances(embeddings, labels)
        
        # Classification evaluation
        logger.info("Computing classification metrics...")
        classification_metrics = self._evaluate_classification(embeddings, labels)
        
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
    
    
    def _generate_embeddings(self, dataset: LacrossePlayerDataset, batch_size: Optional[int] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Generate embeddings for all images in the dataset using efficient batch processing.
        
        This method uses DataLoader for efficient batching and parallel data loading,
        providing significant speedup (typically 10-15x faster) compared to 
        single-image processing while maintaining the same results.
        
        Args:
            dataset: LacrossePlayerDataset instance
            batch_size: Optional batch size override. If None, uses evaluator_config.batch_size
        
        Returns:
            embeddings: Numpy array of embeddings
            labels: List of player labels  
            image_paths: List of image file paths
        """
        # Use provided batch_size or fall back to config
        effective_batch_size = batch_size if batch_size is not None else evaluator_config.batch_size
        
        logger.info(f"Generating embeddings for {len(dataset)} samples using batch processing...")
        logger.info(f"Batch size: {effective_batch_size}, Num workers: {evaluator_config.default_workers}")
        logger.info(f"Model is in eval mode: {not self.model.training}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        
        # Model diagnostics - check if model seems trained
        self._log_model_diagnostics()
        
        # Create DataLoader for efficient batch processing
        eval_dataloader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=False,  # Keep original order for consistent indexing
            num_workers=evaluator_config.default_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=False  # Include all samples
        )
        
        all_embeddings = []
        all_labels = []
        all_image_paths = []
        
        logger.info(f"Processing {len(eval_dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_dataloader):
                try:
                    # Handle different dataset formats
                    if len(batch_data) == 4:
                        # Triplet format: (anchor, positive, negative, labels)
                        anchors, _, _, labels = batch_data
                        images = anchors  # Use anchor images for embedding
                    elif len(batch_data) == 2:
                        # Simple format: (images, labels)
                        images, labels = batch_data
                    else:
                        logger.warning(f"Unexpected batch format with {len(batch_data)} elements in batch {batch_idx}")
                        continue
                    
                    # Move batch to device
                    images = images.to(self.device)
                    
                    # Process entire batch at once - major speedup!
                    batch_embeddings = self.model(images)
                    
                    # Store results
                    all_embeddings.append(batch_embeddings.cpu().numpy())
                    
                    # Convert labels to strings and extend the list
                    batch_labels = [str(label.item()) if torch.is_tensor(label) else str(label) 
                                   for label in labels]
                    all_labels.extend(batch_labels)
                    
                    # Generate image path identifiers
                    batch_start_idx = batch_idx * effective_batch_size
                    batch_image_paths = [f"image_{batch_start_idx + i}" for i in range(len(batch_labels))]
                    all_image_paths.extend(batch_image_paths)
                    
                    # Progress logging every 10 batches
                    if batch_idx % 10 == 0 or batch_idx == len(eval_dataloader) - 1:
                        processed = min((batch_idx + 1) * effective_batch_size, len(dataset))
                        logger.info(f"Processed batch {batch_idx + 1}/{len(eval_dataloader)} - {processed}/{len(dataset)} images")
                        
                        # Debug first batch
                        if batch_idx == 0:
                            logger.info(f"First batch - images shape: {images.shape}, embeddings shape: {batch_embeddings.shape}")
                            logger.info(f"First batch - embeddings stats: min={batch_embeddings.min():.4f}, max={batch_embeddings.max():.4f}, mean={batch_embeddings.mean():.4f}")
                            logger.info(f"First batch - embeddings norm: {torch.norm(batch_embeddings, p=2, dim=1).mean():.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch {batch_idx}: {e}")
                    import traceback
                    logger.warning(f"Full traceback: {traceback.format_exc()}")
                    continue
        
        # Concatenate all embeddings efficiently
        if all_embeddings:
            embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            logger.error("No embeddings generated!")
            return np.array([]), [], []
        
        logger.info(f"Generated {len(embeddings)} embeddings for {len(set(all_labels))} unique players")
        
        # Validate embeddings quality
        self._validate_embeddings(embeddings)
        
        return embeddings, all_labels, all_image_paths
    
    
    def _log_model_diagnostics(self):
        """Log model diagnostics to assess if model is properly trained."""
        # Check if model seems to be trained by looking at some weights
        sample_weights = []
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.numel() > 0:
                sample_weights.extend(param.flatten()[:10].cpu().detach().numpy())
                if len(sample_weights) >= 50:  # Get a reasonable sample
                    break
        
        if sample_weights:
            weight_std = np.std(sample_weights)
            weight_mean = np.mean(sample_weights)
            logger.info(f"Model weight sample stats: mean={weight_mean:.6f}, std={weight_std:.6f}")
            if weight_std < 1e-6:
                logger.warning("⚠️ Model weights have very low variance - model might not be trained!")
            elif weight_std > 2.0:
                logger.warning("⚠️ Model weights have very high variance - might be untrained or corrupted!")
            else:
                logger.info("✓ Model weights seem reasonable")
    
    
    def _validate_embeddings(self, embeddings: np.ndarray):
        """Validate the quality of generated embeddings."""
        if len(embeddings) == 0:
            logger.error("No embeddings generated!")
            return
            
        logger.info(f"Final embeddings array shape: {embeddings.shape}")
        logger.info(f"Embeddings stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, mean={embeddings.mean():.4f}")
        
        # Check for all-zero embeddings
        zero_embeddings = np.sum(np.all(embeddings == 0, axis=1))
        if zero_embeddings > 0:
            logger.warning(f"Found {zero_embeddings} all-zero embeddings out of {len(embeddings)}!")
        
        # Check for NaN embeddings
        nan_embeddings = np.sum(np.any(np.isnan(embeddings), axis=1))
        if nan_embeddings > 0:
            logger.warning(f"Found {nan_embeddings} embeddings with NaN values!")
        
        # Check embedding norms (should be ~1.0 if L2 normalized)
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(f"Embedding norms: mean={norms.mean():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")
        
        if np.any(norms < 1e-8):
            logger.warning(f"Found {np.sum(norms < 1e-8)} embeddings with near-zero norms!")
        
        logger.info("✓ Embedding validation completed")
    

    def _evaluate_distances(self, embeddings: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """
        Evaluate distance-based metrics using vectorized operations for efficiency.
        
        This method computes pairwise Euclidean distances and cosine similarities
        between all embeddings using vectorized operations (torch.cdist and matrix
        multiplication) instead of nested loops. This provides significant speedup
        (typically 10-50x faster) while maintaining numerical accuracy.
        
        Args:
            embeddings: Array of embeddings with shape (n_samples, embedding_dim)
            labels: List of player labels corresponding to each embedding
            
        Returns:
            Dictionary containing distance and similarity statistics:
            - avg_distance_same_player: Average Euclidean distance between same player pairs
            - avg_distance_different_player: Average Euclidean distance between different player pairs
            - avg_similarity_same_player: Average cosine similarity between same player pairs
            - avg_similarity_different_player: Average cosine similarity between different player pairs
            - distance_separation: Difference in average distances (different - same)
            - similarity_separation: Difference in average similarities (same - different)
            - same_player_pairs_count: Number of same player pairs
            - different_player_pairs_count: Number of different player pairs
        """
        logger.info(f"Evaluating distances for {len(embeddings)} embeddings with {len(set(labels))} unique players")
        logger.info(f"Player labels: {set(labels)}")
        
        # Check for NaN embeddings
        nan_count = np.sum(np.isnan(embeddings))
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in embeddings!")
        
        # Convert embeddings to torch tensor for efficient computation
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        labels_array = np.array(labels)
        
        # Vectorized computation of all pairwise distances and similarities
        # Euclidean distances: use torch.cdist for efficiency
        euclidean_distance_matrix = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)
        
        # Cosine similarities: normalize embeddings and compute dot product
        embeddings_norm = F.normalize(embeddings_tensor, p=2, dim=1)
        cosine_similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.T)
        
        # Create same-player mask using broadcasting
        same_player_matrix = (labels_array[:, None] == labels_array[None, :])
        
        # Extract upper triangular part (excluding diagonal) to get unique pairs
        n = len(embeddings)
        upper_tri_mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        
        # Extract pairwise values for unique pairs only
        euclidean_distances = euclidean_distance_matrix[upper_tri_mask]
        cosine_similarities = cosine_similarity_matrix[upper_tri_mask]
        same_player_pairs = same_player_matrix[upper_tri_mask.numpy()]
        
        # Debug first few pairs
        for i in range(min(5, len(same_player_pairs))):
            logger.info(f"Pair {i+1}: same={same_player_pairs[i]}, dist={euclidean_distances[i]:.4f}, sim={cosine_similarities[i]:.4f}")
        
        # Count same vs different player pairs
        same_count = np.sum(same_player_pairs)
        different_count = np.sum(~same_player_pairs)
        logger.info(f"Pair counts: Same player pairs = {same_count}, Different player pairs = {different_count}")
        
        # Compute statistics using boolean indexing
        same_player_mask = same_player_pairs
        different_player_mask = ~same_player_pairs
        
        # Handle case where no same-player pairs exist
        if same_count == 0:
            logger.warning("No same-player pairs found! All images are from different players.")
            avg_distance_same = float('nan')
            avg_similarity_same = float('nan')
        else:
            avg_distance_same = float(euclidean_distances[same_player_mask].mean())
            avg_similarity_same = float(cosine_similarities[same_player_mask].mean())
        
        # Handle case where no different-player pairs exist
        if different_count == 0:
            logger.warning("No different-player pairs found! All images are from the same player.")
            avg_distance_different = float('nan')
            avg_similarity_different = float('nan')
        else:
            avg_distance_different = float(euclidean_distances[different_player_mask].mean())
            avg_similarity_different = float(cosine_similarities[different_player_mask].mean())
        
        # Calculate separations (handle NaN cases)
        if same_count > 0 and different_count > 0:
            distance_separation = avg_distance_different - avg_distance_same
            similarity_separation = avg_similarity_same - avg_similarity_different
        else:
            distance_separation = float('nan')
            similarity_separation = float('nan')
        
        metrics = {
            'avg_distance_same_player': avg_distance_same,
            'avg_distance_different_player': avg_distance_different,
            'avg_similarity_same_player': avg_similarity_same,
            'avg_similarity_different_player': avg_similarity_different,
            'distance_separation': distance_separation,
            'similarity_separation': similarity_separation,
            'same_player_pairs_count': same_count,
            'different_player_pairs_count': different_count
        }
        
        logger.info(f"Distance metrics calculated: same_avg={avg_distance_same:.4f}, diff_avg={avg_distance_different:.4f}")
        
        return metrics
    
    def _evaluate_classification(self, embeddings: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """
        Evaluate classification metrics using similarity threshold.
        """
        logger.info(f"Starting classification evaluation with {len(embeddings)} embeddings")
        logger.info(f"Unique labels: {set(labels)}")
        logger.info(f"Labels distribution: {[(label, labels.count(label)) for label in set(labels)]}")
        
        # Check if embeddings are problematic
        if len(embeddings) == 0:
            logger.error("No embeddings provided for classification evaluation!")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                'f1_score': 0.0, 'roc_auc': 0.0, 'threshold_used': self.threshold
            }
        
        # Check if all embeddings are the same (which would cause zero similarity variance)
        embedding_variance = np.var(embeddings, axis=0)
        logger.info(f"Embedding variance stats: min={embedding_variance.min():.8f}, max={embedding_variance.max():.8f}, mean={embedding_variance.mean():.8f}")
        
        zero_variance_dims = np.sum(embedding_variance < 1e-8)
        if zero_variance_dims > 0:
            logger.warning(f"Found {zero_variance_dims} dimensions with near-zero variance out of {len(embedding_variance)}!")
        
        # Generate pairs and predictions
        y_true = []
        y_pred = []
        y_scores = []
        
        logger.info("Generating pairs for classification...")
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # True label (1 if same player, 0 if different)
                is_same_player = labels[i] == labels[j]
                y_true.append(int(is_same_player))
                
                # Similarity score (cosine similarity)
                emb_i = embeddings[i]
                emb_j = embeddings[j]
                
                # Check for zero embeddings in this pair
                if np.allclose(emb_i, 0, atol=1e-8) or np.allclose(emb_j, 0, atol=1e-8):
                    logger.warning(f"Zero embedding detected in pair ({i}, {j})")
                
                norm_i = np.linalg.norm(emb_i)
                norm_j = np.linalg.norm(emb_j)
                
                if norm_i == 0 or norm_j == 0:
                    logger.warning(f"Zero norm detected in pair ({i}, {j}): norm_i={norm_i:.8f}, norm_j={norm_j:.8f}")
                    similarity = 0.0  # Default similarity for zero-norm vectors
                else:
                    similarity = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                
                y_scores.append(similarity)
                
                # Prediction based on threshold
                y_pred.append(int(similarity > self.threshold))
                
                # Debug first few pairs
                if len(y_true) <= 5:
                    logger.info(f"Pair {len(y_true)}: labels=({labels[i]}, {labels[j]}), same={is_same_player}, "
                              f"similarity={similarity:.6f}, pred={int(similarity > self.threshold)}, threshold={self.threshold}")
        
        logger.info(f"Generated {len(y_true)} pairs for evaluation")
        
        if len(y_true) == 0:
            logger.error("No valid pairs generated for classification!")
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                'f1_score': 0.0, 'roc_auc': 0.0, 'threshold_used': self.threshold
            }
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=np.nan)
        
        # Add detailed debugging information
        logger.info(f"Classification evaluation: {len(y_true)} pairs generated")
        logger.info(f"True labels distribution: Same={sum(y_true)}, Different={len(y_true)-sum(y_true)}")
        logger.info(f"Predictions distribution: Same={sum(y_pred)}, Different={len(y_pred)-sum(y_pred)}")
        
        # Show similarity distribution for same vs different players
        same_similarities = [score for score, true_label in zip(y_scores, y_true) if true_label == 1]
        diff_similarities = [score for score, true_label in zip(y_scores, y_true) if true_label == 0]
        
        if same_similarities:
            logger.info(f"Same player similarities: mean={np.mean(same_similarities):.6f}, "
                       f"min={min(same_similarities):.6f}, max={max(same_similarities):.6f}")
        else:
            logger.warning("No same-player pairs found!")
            
        if diff_similarities:
            logger.info(f"Different player similarities: mean={np.mean(diff_similarities):.6f}, "
                       f"min={min(diff_similarities):.6f}, max={max(diff_similarities):.6f}")
        else:
            logger.warning("No different-player pairs found!")
        
        logger.info(f"Threshold used: {self.threshold}")
        
        # Count true positives, false positives, etc. for debugging
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
        tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
        
        logger.info(f"Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        logger.info(f"Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Debug: Show a few sample predictions vs truth
        logger.info("Sample predictions vs truth (first 10 pairs):")
        for i in range(min(10, len(y_true))):
            logger.info(f"  Pair {i}: similarity={y_scores[i]:.6f}, true={y_true[i]}, pred={y_pred[i]}, threshold={self.threshold}")


        # Always try to find a better threshold, especially if results are poor
        if len(set(y_true)) == 2:  # Only if we have both classes in ground truth
            logger.info("Attempting to find optimal threshold...")
            best_threshold, best_f1 = self._find_optimal_threshold(y_true, y_scores)
            logger.info(f"Optimal threshold found: {best_threshold:.6f} with F1: {best_f1:.6f}")
            
            # Use optimal threshold if it's better than current F1 or if current F1 is very poor
            if best_f1 > f1 or f1 < 0.1:
                y_pred_optimal = [int(score > best_threshold) for score in y_scores]
                
                # Recalculate confusion matrix with optimal threshold
                tp_opt = sum(1 for true, pred in zip(y_true, y_pred_optimal) if true == 1 and pred == 1)
                fp_opt = sum(1 for true, pred in zip(y_true, y_pred_optimal) if true == 0 and pred == 1)
                fn_opt = sum(1 for true, pred in zip(y_true, y_pred_optimal) if true == 1 and pred == 0)
                tn_opt = sum(1 for true, pred in zip(y_true, y_pred_optimal) if true == 0 and pred == 0)
                
                precision = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0.0
                recall = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Recalculate accuracy with optimal threshold
                accuracy = accuracy_score(y_true, y_pred_optimal)
                
                logger.info(f"Using optimal threshold {best_threshold:.6f}")
                logger.info(f"Results with optimal threshold: TP={tp_opt}, FP={fp_opt}, TN={tn_opt}, FN={fn_opt}")
                logger.info(f"Improved metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
                # Update the threshold for future use
                self.threshold = best_threshold
            else:
                logger.info(f"Keeping original threshold {self.threshold:.6f} (F1={f1:.4f} vs optimal F1={best_f1:.4f})")
        else:
            logger.warning("Cannot optimize threshold - only one class present in ground truth")
        
        try:
            auc = roc_auc_score(y_true, y_scores)
            logger.info(f"ROC-AUC calculated: {auc:.6f}")
        except ValueError as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            auc = 0.0  # Handle case where only one class is present
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'threshold_used': self.threshold
        }
    

    def _find_optimal_threshold(self, y_true: List[int], y_scores: List[float]) -> Tuple[float, float]:
        """
        Find the optimal threshold that maximizes F1 score.
        """
        thresholds = np.linspace(min(y_scores), max(y_scores), 50)
        best_f1 = 0.0
        best_threshold = self.threshold
        
        for threshold in thresholds:
            y_pred_temp = [int(score > threshold) for score in y_scores]
            if len(set(y_pred_temp)) == 2:  # Only consider thresholds that produce both classes
                # Manual F1 calculation to avoid sklearn issues
                tp = sum(1 for true, pred in zip(y_true, y_pred_temp) if true == 1 and pred == 1)
                fp = sum(1 for true, pred in zip(y_true, y_pred_temp) if true == 0 and pred == 1)
                fn = sum(1 for true, pred in zip(y_true, y_pred_temp) if true == 1 and pred == 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_temp = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                if f1_temp > best_f1:
                    best_f1 = f1_temp
                    best_threshold = float(threshold)
        
        return best_threshold, best_f1
    

    def _evaluate_ranking(self, embeddings: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """
        Evaluate ranking metrics (Rank-1, Rank-5, mAP).
        """
        logger.info(f"Starting ranking evaluation with {len(embeddings)} embeddings")
        unique_labels = list(set(labels))
        logger.info(f"Unique players for ranking: {len(unique_labels)} ({unique_labels})")
        
        if len(embeddings) < 2:
            logger.warning("Not enough embeddings for ranking evaluation!")
            return {
                'rank_1_accuracy': 0.0,
                'rank_5_accuracy': 0.0,
                'mean_average_precision': 0.0
            }
        
        # For each query, find ranking of correct matches
        rank_1_correct = 0
        rank_5_correct = 0
        average_precisions = []
        
        for query_idx in range(len(embeddings)):
            query_embedding = embeddings[query_idx]
            query_label = labels[query_idx]
            
            # Check for zero query embedding
            if np.allclose(query_embedding, 0, atol=1e-8):
                logger.warning(f"Query {query_idx} has zero embedding!")
            
            # Compute similarities to all other embeddings
            similarities = []
            gallery_labels = []
            
            for gallery_idx in range(len(embeddings)):
                if gallery_idx != query_idx:  # Exclude the query itself
                    gallery_embedding = embeddings[gallery_idx]
                    
                    # Check for zero gallery embedding
                    if np.allclose(gallery_embedding, 0, atol=1e-8):
                        logger.warning(f"Gallery {gallery_idx} has zero embedding!")
                    
                    # Calculate cosine similarity
                    query_norm = np.linalg.norm(query_embedding)
                    gallery_norm = np.linalg.norm(gallery_embedding)
                    
                    if query_norm == 0 or gallery_norm == 0:
                        similarity = 0.0  # Default similarity for zero-norm vectors
                    else:
                        similarity = np.dot(query_embedding, gallery_embedding) / (query_norm * gallery_norm)
                    
                    similarities.append(similarity)
                    gallery_labels.append(labels[gallery_idx])
            
            if not similarities:
                logger.warning(f"No gallery items for query {query_idx}")
                continue
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_labels = [gallery_labels[i] for i in sorted_indices]
            sorted_similarities = [similarities[i] for i in sorted_indices]
            
            # Debug first few queries
            if query_idx < 3:
                logger.info(f"Query {query_idx} (label={query_label}):")
                logger.info(f"  Top 5 matches: {[(sorted_labels[i], sorted_similarities[i]) for i in range(min(5, len(sorted_labels)))]}")
                correct_matches_in_top5 = [i for i in range(min(5, len(sorted_labels))) if sorted_labels[i] == query_label]
                logger.info(f"  Correct matches in top 5: {correct_matches_in_top5}")
            
            # Check rank-1 and rank-5
            if len(sorted_labels) > 0 and sorted_labels[0] == query_label:
                rank_1_correct += 1
            
            if len(sorted_labels) >= 5 and query_label in sorted_labels[:5]:
                rank_5_correct += 1
            elif len(sorted_labels) < 5 and query_label in sorted_labels:
                rank_5_correct += 1
            
            # Compute Average Precision for this query
            ap = self._compute_average_precision(sorted_labels, query_label)
            average_precisions.append(ap)
            
            # Debug AP calculation for first few queries
            if query_idx < 3:
                logger.info(f"  Average Precision: {ap:.6f}")
        
        # Compute final metrics
        rank_1_accuracy = rank_1_correct / len(embeddings) if len(embeddings) > 0 else 0
        rank_5_accuracy = rank_5_correct / len(embeddings) if len(embeddings) > 0 else 0
        mean_ap = np.mean(average_precisions) if average_precisions else 0
        
        logger.info(f"Ranking results: Rank-1={rank_1_correct}/{len(embeddings)} ({rank_1_accuracy:.6f})")
        logger.info(f"Ranking results: Rank-5={rank_5_correct}/{len(embeddings)} ({rank_5_accuracy:.6f})")
        logger.info(f"Mean AP: {mean_ap:.6f} (from {len(average_precisions)} queries)")
        
        return {
            'rank_1_accuracy': float(rank_1_accuracy),
            'rank_5_accuracy': float(rank_5_accuracy),
            'mean_average_precision': float(mean_ap)
        }
    
    def _compute_average_precision(self, sorted_labels: List[str], query_label: str) -> float:
        """
        Compute Average Precision for a single query.
        """
        if not sorted_labels:
            return 0.0
        
        # Find positions of correct matches
        correct_matches = [i for i, label in enumerate(sorted_labels) if label == query_label]
        
        if not correct_matches:
            return 0.0
        
        # Compute precision at each correct match
        precisions = []
        for rank in correct_matches:
            precision_at_rank = (len([i for i in correct_matches if i <= rank])) / (rank + 1)
            precisions.append(precision_at_rank)
        
        return float(np.mean(precisions)) if precisions else 0.0
    
    
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
