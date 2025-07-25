import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

from core.train.dataset import LacrossePlayerDataset
from config.all_config import wandb_config
from core.train.wandb_logger import wandb_logger

logger = logging.getLogger(__name__)


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
                 threshold: float = 0.5, k_folds: int = 5):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained Siamese network model
            device: Device to run evaluation on
            threshold: Similarity threshold for binary classification
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
                             use_validation_split: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive evaluation including all metrics.
        
        Args:
            dataset_path: Path to the dataset directory (should contain train/ and val/ folders)
            use_validation_split: If True, use existing val/ folder; if False, create random split
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        if use_validation_split:
            # Use existing train/val split
            train_dir = os.path.join(dataset_path, "train")
            val_dir = os.path.join(dataset_path, "val")
            
            if not os.path.exists(val_dir):
                logger.warning(f"Validation directory not found: {val_dir}")
                logger.info("Falling back to random split from training data")
                use_validation_split = False
            else:
                logger.info(f"Using existing validation split:")
                logger.info(f"  Train dir: {train_dir}")
                logger.info(f"  Val dir: {val_dir}")
                
                # Create dataset from validation directory
                from config.transforms import get_transforms
                val_transforms = get_transforms('validation')  # Use validation transforms
                val_dataset = LacrossePlayerDataset(
                    image_dir=val_dir,
                    transform=val_transforms
                )
                
                # Also create training dataset for cross-validation
                train_transforms = get_transforms('training')
                train_dataset = LacrossePlayerDataset(
                    image_dir=train_dir,
                    transform=train_transforms
                )
        
        if not use_validation_split:
            # Fallback: create random split from dataset (old behavior)
            logger.info("Creating random split from provided dataset")
            dataset = LacrossePlayerDataset(image_dir=dataset_path)
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
                    
                    image = image.unsqueeze(0).to(self.device)
                    
                    # Get embedding from model
                    embedding = self.model(image)  # Use regular forward method
                    embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
                    
                    embeddings.append(embedding.cpu().numpy().flatten())
                    labels.append(str(label.item()) if torch.is_tensor(label) else str(label))
                    
                    # Create a simple image path identifier
                    image_paths.append(f"image_{i}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process image {i}: {e}")
                    continue
        
        return np.array(embeddings), labels, image_paths
    
    def _evaluate_distances(self, embeddings: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """
        Evaluate distance-based metrics.
        """
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
                same_player_pairs.append(labels[i] == labels[j])
        
        # Convert to numpy arrays
        euclidean_distances = np.array(euclidean_distances)
        cosine_similarities = np.array(cosine_similarities)
        same_player_pairs = np.array(same_player_pairs)
        
        # Compute statistics
        same_player_mask = same_player_pairs
        different_player_mask = ~same_player_pairs
        
        metrics = {
            'avg_distance_same_player': float(np.mean(euclidean_distances[same_player_mask])),
            'avg_distance_different_player': float(np.mean(euclidean_distances[different_player_mask])),
            'avg_similarity_same_player': float(np.mean(cosine_similarities[same_player_mask])),
            'avg_similarity_different_player': float(np.mean(cosine_similarities[different_player_mask])),
            'distance_separation': float(np.mean(euclidean_distances[different_player_mask]) - 
                                       np.mean(euclidean_distances[same_player_mask])),
            'similarity_separation': float(np.mean(cosine_similarities[same_player_mask]) - 
                                         np.mean(cosine_similarities[different_player_mask]))
        }
        
        return metrics
    
    def _evaluate_classification(self, embeddings: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """
        Evaluate classification metrics using similarity threshold.
        """
        # Generate pairs and predictions
        y_true = []
        y_pred = []
        y_scores = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # True label (1 if same player, 0 if different)
                y_true.append(int(labels[i] == labels[j]))
                
                # Similarity score (cosine similarity)
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                y_scores.append(similarity)
                
                # Prediction based on threshold
                y_pred.append(int(similarity > self.threshold))
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.0  # Handle case where only one class is present
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'threshold_used': self.threshold
        }
    
    def _evaluate_ranking(self, embeddings: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """
        Evaluate ranking metrics (Rank-1, Rank-5, mAP).
        """
        unique_labels = list(set(labels))
        
        # For each query, find ranking of correct matches
        rank_1_correct = 0
        rank_5_correct = 0
        average_precisions = []
        
        for query_idx in range(len(embeddings)):
            query_embedding = embeddings[query_idx]
            query_label = labels[query_idx]
            
            # Compute similarities to all other embeddings
            similarities = []
            gallery_labels = []
            
            for gallery_idx in range(len(embeddings)):
                if gallery_idx != query_idx:  # Exclude the query itself
                    similarity = np.dot(query_embedding, embeddings[gallery_idx]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embeddings[gallery_idx])
                    )
                    similarities.append(similarity)
                    gallery_labels.append(labels[gallery_idx])
            
            # Sort by similarity (descending)
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_labels = [gallery_labels[i] for i in sorted_indices]
            
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
        
        # Compute final metrics
        rank_1_accuracy = rank_1_correct / len(embeddings) if len(embeddings) > 0 else 0
        rank_5_accuracy = rank_5_correct / len(embeddings) if len(embeddings) > 0 else 0
        mean_ap = np.mean(average_precisions) if average_precisions else 0
        
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
        
        # Save JSON results
        results_file = os.path.join(save_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
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
        report.append(f"\nDataset Information:")
        report.append(f"  Total samples: {dataset_info['total_samples']}")
        report.append(f"  Test samples: {dataset_info['test_samples']}")
        report.append(f"  Number of players: {dataset_info['num_players']}")
        report.append(f"  Test players: {dataset_info['test_players']}")
        
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
