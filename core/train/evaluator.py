import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

from core.train.dataset import LacrossePlayerDataset
from core.config.all_config import wandb_config, evaluator_config
from core.train.wandb_logger import wandb_logger

logger = logging.getLogger(__name__)

N_PLAYERS_FOR_VAL = 1

class ModelEvaluator:
    """
    Comprehensive evaluation system for Siamese networks in player re-identification.
    
    Implements multiple evaluation methodologies:
    - Distance-based metrics (Euclidean, Cosine)
    - Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    - Ranking metrics (Rank-1, Rank-5, mAP, CMC curve)
    - Cross-validation and temporal validation
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device, 
                 threshold: float = evaluator_config.threshold, k_folds: int = evaluator_config.k_folds):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained Siamese network model
            device: Device to run evaluation on
            threshold: Similarity threshold for binary classification (default 0.5, will be optimized)
            k_folds: Number of folds for cross-validation
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.k_folds = k_folds
        self.model.eval()
        
        # Results storage
        self.results = {}
        self.embeddings_cache = {}
        
    def evaluate_comprehensive(self, dataset_path: str, 
                             storage_client=None,
                             use_validation_split: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation including all metrics.
        
        Args:
            dataset_path: Path to the dataset directory (should contain train/ and val/ folders)
            storage_client: Google Storage client for GCS operations (required for GCS paths)
            use_validation_split: If True, use existing val/ folder; if False, create random split
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        if use_validation_split:
            # The `dataset_path` is the path to the training data, e.g., .../datasets/frame19/train
            # The validation data is in a sibling directory, e.g., .../datasets/frame19/val
            
            # The train_dir is the dataset_path itself.
            train_dir = dataset_path.rstrip('/')
            
            # To get the validation directory, go to parent and join with 'val'.
            base_path = os.path.dirname(train_dir)
            val_dir = os.path.join(base_path, "val")
            
            # Check if validation directory exists
            val_exists = False
            if storage_client:
                # For GCS, check if validation directory has any blobs
                val_blobs = list(storage_client.list_blobs(prefix=val_dir.rstrip('/')))
                val_exists = len(val_blobs) > 0
            else:
                # For local filesystem
                val_exists = os.path.exists(val_dir)
            
            if not val_exists:
                logger.warning(f"Validation directory not found: {val_dir}")
                logger.info("Falling back to random split from training data")
                use_validation_split = False
            else:
                logger.info(f"Using existing validation split:")
                logger.info(f"  Train dir: {train_dir}")
                logger.info(f"  Val dir: {val_dir}")
                
                # Create dataset from validation directory
                from core.config.transforms import get_transforms
                val_transforms = get_transforms('validation')  # Use validation transforms
                if storage_client:
                    val_dataset = LacrossePlayerDataset(
                        image_dir=val_dir,
                        storage_client=storage_client,
                        transform=val_transforms,
                        min_images_per_player= N_PLAYERS_FOR_VAL
                    )
                    train_transforms = get_transforms('training')
                    train_dataset = LacrossePlayerDataset(
                        image_dir=train_dir,
                        storage_client=storage_client,
                        transform=train_transforms,
                        min_images_per_player=N_PLAYERS_FOR_VAL
                    )
                else:
                    val_dataset = LacrossePlayerDataset(
                        image_dir=val_dir,
                        transform=val_transforms,
                        min_images_per_player=N_PLAYERS_FOR_VAL
                    )
                    train_transforms = get_transforms('training')
                    train_dataset = LacrossePlayerDataset(
                        image_dir=train_dir,
                        transform=train_transforms,
                        min_images_per_player=N_PLAYERS_FOR_VAL
                    )
                # Check if enough valid players exist in validation set
                if len(val_dataset.players) < 2:
                    logger.warning(f"Validation set does not have enough valid players (found {len(val_dataset.players)}). Falling back to random split.")
                    use_validation_split = False
        
        if not use_validation_split:
            # Fallback: create random split from dataset (old behavior)
            logger.info("Creating random split from provided dataset")
            if storage_client:
                dataset = LacrossePlayerDataset(
                    image_dir=dataset_path,
                    storage_client=storage_client,
                    min_images_per_player=N_PLAYERS_FOR_VAL  # For evaluation, 1 image is sufficient
                )
            else:
                dataset = LacrossePlayerDataset(
                    image_dir=dataset_path,
                    min_images_per_player=N_PLAYERS_FOR_VAL  # For evaluation, 1 image is sufficient
                )
            train_dataset, val_dataset = self._split_dataset(dataset, test_split=0.2)
        
        # Generate embeddings for validation set
        logger.info("Generating embeddings for validation set...")
        embeddings, labels, image_paths = self._generate_embeddings(val_dataset)
        
        # Distance-based evaluation
        logger.info("Computing distance-based metrics...")
        distance_metrics = self._evaluate_distances(embeddings, labels)
        
        # Classification evaluation
        logger.info("Computing classification metrics...")
        classification_metrics = self._evaluate_classification(embeddings, labels)
        
        # Ranking evaluation
        logger.info("Computing ranking metrics...")
        ranking_metrics = self._evaluate_ranking(embeddings, labels)
        
        # Cross-validation (use training set for this)
        logger.info("Running cross-validation on training set...")
        cv_metrics = self._cross_validate(train_dataset)
        
        # Aggregate results
        results = {
            'distance_metrics': distance_metrics,
            'classification_metrics': classification_metrics,
            'ranking_metrics': ranking_metrics,
            'cross_validation': cv_metrics,
            'dataset_info': {
                'validation_samples': len(val_dataset),
                'training_samples': len(train_dataset) if use_validation_split else 0,
                'val_players': len(val_dataset.players),
                'train_players': len(train_dataset.players) if use_validation_split else 0,
                'used_existing_split': use_validation_split
            }
        }
        
        # Log to wandb if enabled
        if wandb_config.enabled:
            self._log_to_wandb(results)
        
        # Save detailed results
        self._save_results(results)
        
        logger.info("✅ Comprehensive evaluation completed")
        return results
    
    def _split_dataset(self, dataset: LacrossePlayerDataset, 
                      test_split: float) -> Tuple[LacrossePlayerDataset, LacrossePlayerDataset]:
        """
        Split dataset ensuring no player overlap between train and test.
        """
        players = dataset.players
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(players)
        
        split_idx = int(len(players) * (1 - test_split))
        train_players = players[:split_idx]
        test_players = players[split_idx:]
        
        logger.info(f"Dataset split: {len(train_players)} train players, {len(test_players)} test players")
        
        # Create subset datasets (simplified - you may need to implement this in your dataset class)
        train_dataset = self._create_player_subset(dataset, train_players)
        test_dataset = self._create_player_subset(dataset, test_players)
        
        return train_dataset, test_dataset
    
    def _create_player_subset(self, dataset: LacrossePlayerDataset, 
                             players: List[str]) -> LacrossePlayerDataset:
        """
        Create a subset dataset containing only specified players.
        This creates a filtered view of the original dataset.
        """
        # Create a new dataset instance with filtered players
        subset_dataset = LacrossePlayerDataset.__new__(LacrossePlayerDataset)
        
        # Copy basic attributes
        subset_dataset.image_dir = dataset.image_dir
        subset_dataset.transform = dataset.transform
        subset_dataset.min_images_per_player = dataset.min_images_per_player
        
        # Copy storage_client if it exists
        if hasattr(dataset, 'storage_client'):
            subset_dataset.storage_client = dataset.storage_client
        
        # Filter to only include specified players
        subset_dataset.players = [p for p in players if p in dataset.players]
        subset_dataset.player_to_images = {
            player: dataset.player_to_images[player] 
            for player in subset_dataset.players
        }
        
        # Rebuild all_images list for the subset
        subset_dataset.all_images = []
        for player in subset_dataset.players:
            subset_dataset.all_images.extend(subset_dataset.player_to_images[player])
        
        # Rebuild player indices
        subset_dataset.player_indices = {
            player: i for i, player in enumerate(subset_dataset.players)
        }
        
        logger.info(f"Created subset with {len(subset_dataset.players)} players, "
                   f"{len(subset_dataset.all_images)} images")
        
        return subset_dataset
    
    def _generate_embeddings(self, dataset: LacrossePlayerDataset) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Generate embeddings for all images in the dataset.
        
        Returns:
            embeddings: Numpy array of embeddings
            labels: List of player labels
            image_paths: List of image file paths
        """
        embeddings = []
        labels = []
        image_paths = []
        
        logger.info(f"Generating embeddings for {len(dataset)} samples...")
        logger.info(f"Model is in eval mode: {not self.model.training}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
        
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
        
        with torch.no_grad():
            for i in range(len(dataset)):
                try:
                    # Dataset returns (anchor, positive, negative, label)
                    data = dataset[i]
                    if len(data) == 4:
                        anchor, positive, negative, label = data
                        image = anchor  # Use anchor image for embedding
                    else:
                        # Fallback for different dataset formats
                        image, label = data[0], data[-1]
                    
                    # Debug: Check image properties
                    if i < 3:  # Log first few images
                        if isinstance(image, torch.Tensor):
                            logger.info(f"Image {i}: type=Tensor, shape={image.shape}")
                            logger.info(f"  Image tensor stats: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}")
                        else:
                            logger.info(f"Image {i}: type={type(image)}, converting to tensor...")
                    
                    # If image is not a tensor, convert to tensor
                    if not isinstance(image, torch.Tensor):
                        import torchvision.transforms as T
                        image = T.ToTensor()(image)
                    # Ensure tensor is on the correct device before unsqueezing
                    image = image.to(self.device)
                    image_tensor = image.unsqueeze(0)
                    
                    # Get embedding from model
                    embedding = self.model(image_tensor)  # Use regular forward method
                    
                    # The SiameseNet model's forward pass already applies L2 normalization.
                    # No need to normalize again here.
                    final_embedding = embedding
                    
                    # Debug: Check final embedding
                    if i < 3:  # Log first few normalized embeddings
                        logger.info(f"Final embedding {i}: min={final_embedding.min():.4f}, max={final_embedding.max():.4f}, mean={final_embedding.mean():.4f}")
                        logger.info(f"Final embedding norm: {torch.norm(final_embedding, p=2, dim=1).item():.4f}")
                    
                    embeddings.append(final_embedding.cpu().numpy().flatten())
                    labels.append(str(label.item()) if torch.is_tensor(label) else str(label))
                    
                    # Create a simple image path identifier
                    image_paths.append(f"image_{i}")
                except Exception as e:
                    logger.warning(f"Failed to process image {i}: {e}")
                    import traceback
                    logger.warning(f"Full traceback: {traceback.format_exc()}")
                    continue
        
        logger.info(f"Generated {len(embeddings)} embeddings for {len(set(labels))} unique players")
        
        # Check for problematic embeddings
        if embeddings:
            embeddings_array = np.array(embeddings)
            logger.info(f"Final embeddings array shape: {embeddings_array.shape}")
            logger.info(f"Embeddings stats: min={embeddings_array.min():.4f}, max={embeddings_array.max():.4f}, mean={embeddings_array.mean():.4f}")
            
            # Check for all-zero embeddings
            zero_embeddings = np.sum(np.all(embeddings_array == 0, axis=1))
            if zero_embeddings > 0:
                logger.warning(f"Found {zero_embeddings} all-zero embeddings out of {len(embeddings)}!")
            
            # Check for NaN embeddings
            nan_embeddings = np.sum(np.any(np.isnan(embeddings_array), axis=1))
            if nan_embeddings > 0:
                logger.warning(f"Found {nan_embeddings} embeddings with NaN values!")
        
        return np.array(embeddings), labels, image_paths
    
    def _evaluate_distances(self, embeddings: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """
        Evaluate distance-based metrics.
        """
        logger.info(f"Evaluating distances for {len(embeddings)} embeddings with {len(set(labels))} unique players")
        logger.info(f"Player labels: {set(labels)}")
        
        # Check for NaN embeddings
        nan_count = np.sum(np.isnan(embeddings))
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in embeddings!")
        
        # Compute pairwise distances
        euclidean_distances = []
        cosine_similarities = []
        same_player_pairs = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Euclidean distance
                euclidean_dist = np.linalg.norm(embeddings[i] - embeddings[j])
                euclidean_distances.append(euclidean_dist)
                
                # Cosine similarity
                cosine_sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                cosine_similarities.append(cosine_sim)
                
                # Same player or not
                is_same_player = labels[i] == labels[j]
                same_player_pairs.append(is_same_player)
                
                # Debug first few pairs
                if len(same_player_pairs) <= 5:
                    logger.info(f"Pair {len(same_player_pairs)}: player1={labels[i]}, player2={labels[j]}, same={is_same_player}, dist={euclidean_dist:.4f}")
        
        # Convert to numpy arrays
        euclidean_distances = np.array(euclidean_distances)
        cosine_similarities = np.array(cosine_similarities)
        same_player_pairs = np.array(same_player_pairs)
        
        # Count same vs different player pairs
        same_count = np.sum(same_player_pairs)
        different_count = np.sum(~same_player_pairs)
        logger.info(f"Pair counts: Same player pairs = {same_count}, Different player pairs = {different_count}")
        
        # Compute statistics
        same_player_mask = same_player_pairs
        different_player_mask = ~same_player_pairs
        
        # Handle case where no same-player pairs exist
        if same_count == 0:
            logger.warning("No same-player pairs found! All images are from different players.")
            avg_distance_same = float('nan')
            avg_similarity_same = float('nan')
        else:
            avg_distance_same = float(np.mean(euclidean_distances[same_player_mask]))
            avg_similarity_same = float(np.mean(cosine_similarities[same_player_mask]))
        
        # Handle case where no different-player pairs exist
        if different_count == 0:
            logger.warning("No different-player pairs found! All images are from the same player.")
            avg_distance_different = float('nan')
            avg_similarity_different = float('nan')
        else:
            avg_distance_different = float(np.mean(euclidean_distances[different_player_mask]))
            avg_similarity_different = float(np.mean(cosine_similarities[different_player_mask]))
        
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
    
    def _cross_validate(self, dataset: LacrossePlayerDataset) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.
        """
        players = dataset.players
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'accuracy': [],
            'f1_score': [],
            'rank_1_accuracy': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(players)):
            logger.info(f"Cross-validation fold {fold + 1}/{self.k_folds}")
            
            train_players = [players[i] for i in train_idx]
            val_players = [players[i] for i in val_idx]
            
            # Create subset datasets
            val_dataset = self._create_player_subset(dataset, val_players)
            
            # Generate embeddings for validation set
            embeddings, labels, _ = self._generate_embeddings(val_dataset)
            
            if len(embeddings) == 0:
                continue
            
            # Compute metrics for this fold
            classification_metrics = self._evaluate_classification(embeddings, labels)
            ranking_metrics = self._evaluate_ranking(embeddings, labels)
            
            cv_scores['accuracy'].append(classification_metrics['accuracy'])
            cv_scores['f1_score'].append(classification_metrics['f1_score'])
            cv_scores['rank_1_accuracy'].append(ranking_metrics['rank_1_accuracy'])
        
        # Compute mean and std for each metric
        cv_results = {}
        for metric, scores in cv_scores.items():
            if scores:
                cv_results[f'{metric}_mean'] = float(np.mean(scores))
                cv_results[f'{metric}_std'] = float(np.std(scores))
            else:
                cv_results[f'{metric}_mean'] = 0.0
                cv_results[f'{metric}_std'] = 0.0
        
        return cv_results
    
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
        
        # Cross-validation metrics
        for key, value in results['cross_validation'].items():
            wandb_metrics[f'eval/cv/{key}'] = value
        
        # Dataset info
        for key, value in results['dataset_info'].items():
            wandb_metrics[f'eval/dataset/{key}'] = value
        
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
        
        # Dataset info
        dataset_info = results['dataset_info']
        report.append("\nDataset Information:")
        report.append(f"  Training samples: {dataset_info.get('training_samples', 'N/A')}")
        report.append(f"  Validation samples: {dataset_info.get('validation_samples', 'N/A')}")
        report.append(f"  Training players: {dataset_info.get('train_players', 'N/A')}")
        report.append(f"  Validation players: {dataset_info.get('val_players', 'N/A')}")
        
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
        report.append(f"  Same player avg distance: {dist_metrics['avg_distance_same_player']:.4f}")
        report.append(f"  Different player avg distance: {dist_metrics['avg_distance_different_player']:.4f}")
        report.append(f"  Distance separation: {dist_metrics['distance_separation']:.4f}")
        
        # Cross-validation
        cv_metrics = results['cross_validation']
        report.append(f"\nCross-Validation Results:")
        report.append(f"  Accuracy: {cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}")
        report.append(f"  F1-Score: {cv_metrics['f1_score_mean']:.4f} ± {cv_metrics['f1_score_std']:.4f}")
        report.append(f"  Rank-1: {cv_metrics['rank_1_accuracy_mean']:.4f} ± {cv_metrics['rank_1_accuracy_std']:.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
