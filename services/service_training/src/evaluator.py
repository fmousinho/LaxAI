import json
import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config.transforms import get_transforms
from dataset import LacrossePlayerDataset
from wandb_logger import wandb_logger

from shared_libs.config.all_config import evaluator_config, wandb_config

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
        
        # Ensure model is on the correct device
        current_device = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device('cpu')
        if str(current_device) != str(device):
            logger.info(f"Moving evaluation model from {current_device} to {device}")
            self.model = self.model.to(device)
        
        # Log memory usage after model setup
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Model loaded for evaluation - Memory: {memory_mb:.1f}MB")

    def evaluate_comprehensive(self, dataset, **eval_kwargs) -> Dict[str, Any]:
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
        embs, labels, image_paths  = self._generate_embeddings(dataset, **eval_kwargs)

        # Defensive: if no embeddings were generated (empty dataset or
        # mocked dataset returning zero samples), return an empty results
        # dict rather than proceeding and hitting concatenation/shape ops.
        if embs is None or (hasattr(embs, 'shape') and embs.shape[0] == 0):
            logger.warning("No embeddings generated for dataset; skipping metric computations and returning empty results")
            empty_results = {
                'distance_metrics': {},
                'classification_metrics': {},
                'ranking_metrics': {},
            }
            # Save results as-is (no-op if backend expects files)
            try:
                self._save_results(empty_results)
            except Exception:
                logger.debug("_save_results failed for empty results; continuing")
            logger.info("✅ Comprehensive evaluation completed (empty dataset)")
            return empty_results

        # Decide whether to compute full pairwise arrays (may be huge) or
        # stream statistics/samples for large datasets. Streaming avoids
        # concatenating large tensors (O(N^2)) which can blow up RAM.
        n = embs.shape[0]
        pairwise_threshold = getattr(evaluator_config, 'batched_ranking_threshold', 5000)
        use_batched = n > pairwise_threshold

        if use_batched:
            logger.info(f"Dataset large (n={n}), computing pairwise metrics in streaming mode to save memory (threshold={pairwise_threshold})")
            # Streaming computation: exact averages for distances/similarities,
            # approximate classification scores via reservoir sampling capped by
            # `pairwise_sample_cap` to bound memory usage during threshold search.
            sample_cap = getattr(evaluator_config, 'pairwise_sample_cap', 1000000)
            distance_metrics, classification_metrics = self._stream_pairwise_metrics(
                embs, labels, batch_size=getattr(evaluator_config, 'pairwise_batch_size', 4096),
                sample_cap=sample_cap
            )
            sims = None
            dists = None
            labels_eq = None
        else:
            sims, dists, labels_eq = self._compute_pairwise_batches(
                embs,
                labels,
                batch_size=getattr(evaluator_config, 'pairwise_batch_size', 4096),
                compute_distances=True
            )

            # Distance-based evaluation
            logger.info("Computing distance-based metrics...")
            distance_metrics = self._evaluate_distances(sims, dists, labels_eq)

            # Classification evaluation
            logger.info("Computing classification metrics...")
            classification_metrics = self._evaluate_classification(sims, labels_eq)

        # Decide whether to build a full NxN similarity matrix or use batched ranking
        n = embs.shape[0]
        threshold = getattr(evaluator_config, 'batched_ranking_threshold', 5000)
        use_batched = n > threshold

        sims_matrix = None
        if use_batched:
            logger.info(f"Dataset large (n={n}), using batched ranking (threshold={threshold})")
        else:
            logger.info("Building full similarity matrix for ranking/recall...")
            device = self.device
            X = torch.tensor(embs, dtype=torch.float32, device=device)
            X_norm = torch.nn.functional.normalize(X, dim=1)
            sims_matrix = torch.mm(X_norm, X_norm.T)

        # Ranking evaluation
        logger.info("Computing ranking metrics...")
        ranking_metrics = self._evaluate_ranking(
            sims_matrix if sims_matrix is not None else embs,
            labels,
            use_batched=use_batched,
            chunk_size=getattr(evaluator_config, 'ranking_chunk_size', 1024)
        )

        # Compute recall@k metrics and merge
        try:
            recall_metrics = self.compute_recall_at_k(
                sims_matrix=sims_matrix if sims_matrix is not None else None,
                labels=labels,
                ks=(1, 5, 10),
                use_batched=use_batched,
                chunk_size=getattr(evaluator_config, 'ranking_chunk_size', 1024)
            )
            ranking_metrics.update(recall_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute recall@k metrics: {e}")

        # Aggregate results
        results = {
            'distance_metrics': distance_metrics,
            'classification_metrics': classification_metrics,
            'ranking_metrics': ranking_metrics,
        }

        # Save detailed results
        self._save_results(results)

        logger.info("✅ Comprehensive evaluation completed")
        
        # Clear large data structures to free memory
        try:
            del embs, labels, image_paths
            if 'sims' in locals() and sims is not None:
                del sims
            if 'dists' in locals() and dists is not None:
                del dists
            if 'labels_eq' in locals() and labels_eq is not None:
                del labels_eq
            if 'sims_matrix' in locals() and sims_matrix is not None:
                del sims_matrix
            import gc
            gc.collect()
        except Exception as cleanup_error:
            logger.debug(f"Evaluation data cleanup warning: {cleanup_error}")
        
        return results


    

    def _generate_embeddings(self, dataset: LacrossePlayerDataset, **eval_kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate embeddings for all images in the dataset in batches.

        Args:
            dataset: LacrossePlayerDataset
            batch_size: batch size for embedding generation (reduced default for memory safety)

        Returns:
            embeddings: Numpy array of embeddings
            labels: Numpy array of corresponding player labels
            image_paths: List of image identifiers.
        """
        from torch.utils.data import DataLoader

        batch_size = eval_kwargs.get("batch_size", evaluator_config.emb_batch_size)
        num_workers = eval_kwargs.get("num_workers", evaluator_config.number_of_workers)
        prefetch_factor = eval_kwargs.get("prefetch_factor", evaluator_config.prefetch_factor)

        logger.info(f"Generating embeddings for {len(dataset)} samples in batches of {batch_size}...")

        # Log initial memory state
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory before embedding generation: {initial_memory:.1f}MB")

        logger.info(f"Using DataLoader with batch_size={batch_size}, num_workers={num_workers}, prefetch_factor={prefetch_factor}")

        # Ensure eval mode and no gradients
        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_paths = []

        if num_workers == 0:
            prefetch_factor = None  # prefetch_factor should only be used with multiple workers


        # Wrap dataset in DataLoader for batching
        # Use minimal workers and no pin_memory to reduce CPU memory usage
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,       # Single-threaded to prevent memory explosion
            pin_memory=False,    # Disable to save CPU memory
            prefetch_factor=prefetch_factor
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Check for cancellation every batch
                if self.stop_callback and self.stop_callback():
                    logger.info(f"Embedding generation cancelled at batch {batch_idx}")
                    break

                # Dataset returns could vary depending on your format
                if len(batch) == 4:  
                    images, _, _, labels = batch
                else:
                    images, labels = batch[0], batch[-1]

                # Move to device
                images = images.to(self.device, non_blocking=True)

                # Forward pass
                embeddings = self.model(images)  # already normalized

                # Move to CPU numpy and explicitly delete GPU tensors
                emb_cpu = embeddings.cpu().numpy()
                all_embeddings.append(emb_cpu)
                
                # Keep labels as numeric values for evaluation
                numeric_labels = []
                for l in labels:
                    if torch.is_tensor(l):
                        numeric_labels.append(l.item())
                    elif isinstance(l, (int, float)):
                        numeric_labels.append(l)
                    else:
                        # Try to convert string to numeric if possible
                        try:
                            numeric_labels.append(float(l))
                        except (ValueError, TypeError):
                            # If conversion fails, use hash of string as numeric ID
                            numeric_labels.append(hash(str(l)) % 1000000)
                
                all_labels.extend(numeric_labels)
                all_paths.extend([f"image_{batch_idx * batch_size + i}" for i in range(len(labels))])

                # Clear GPU memory after each batch
                del images, embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Log progress every batch for large datasets
                processed = batch_idx * batch_size + len(labels)
                
                # Get memory usage
                memory_info = ""
                if PSUTIL_AVAILABLE:
                    # RAM usage
                    process = psutil.Process()
                    ram_mb = process.memory_info().rss / 1024 / 1024
                    memory_info = f" | RAM: {ram_mb:.1f}MB"
                
                # GPU memory if available
                if torch.cuda.is_available():
                    gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                    memory_info += f" | GPU: {gpu_mb:.1f}MB allocated, {gpu_reserved_mb:.1f}MB reserved"
                
                logger.info(f"Processed {processed}/{len(dataset)} images (batch {batch_idx + 1}){memory_info}")

        # Stack results; if no batches were processed return empty arrays
        if not all_embeddings:
            # No samples processed; return empty structures (shape (0, 0) for
            # embeddings so downstream short-circuit in evaluate_comprehensive
            # can detect zero rows).
            embeddings_array = np.empty((0, 0), dtype=np.float32)
            labels_array = np.array([], dtype=np.int64)
            logger.info("No embeddings were produced by the DataLoader (0 samples)")
            return embeddings_array, labels_array, all_paths

        embeddings_array = np.vstack(all_embeddings)
        labels_array = np.array(all_labels)

        logger.info(f"Generated {embeddings_array.shape[0]} embeddings for {len(set(labels_array))} unique players")
        # Protect mean() call when there are zero elements
        if embeddings_array.size > 0:
            logger.info(f"Embeddings shape: {embeddings_array.shape}, mean={embeddings_array.mean():.4f}")
        else:
            logger.info(f"Embeddings shape: {embeddings_array.shape}")

        # Log final memory state
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = final_memory - initial_memory
            logger.info(f"Memory after embedding generation: {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")

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
            sims: torch.Tensor, shape [N_pairs] with cosine similarities
            dists: torch.Tensor or None
            labels_eq: torch.BoolTensor, shape [N_pairs]
        """
        
        device = self.device

        X = torch.tensor(embeddings, dtype=torch.float32, device=device)
        
        # Ensure labels are numeric and convert to tensor
        if isinstance(labels, np.ndarray):
            if labels.dtype.kind in ['U', 'S']:  # Unicode or byte string
                # Convert string labels to numeric IDs
                unique_labels = np.unique(labels)
                label_to_id = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_id[label] for label in labels])
            else:
                numeric_labels = labels.astype(np.int64)
        elif isinstance(labels, list):
            if all(isinstance(l, str) for l in labels):
                # Convert string labels to numeric IDs
                unique_labels = list(set(labels))
                label_to_id = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_id[label] for label in labels])
            else:
                numeric_labels = np.array(labels, dtype=np.int64)
        else:
            numeric_labels = np.array(labels, dtype=np.int64)
            
        y = torch.tensor(numeric_labels, dtype=torch.long, device=device)

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


    def _stream_pairwise_metrics(self, embeddings: np.ndarray, labels: np.ndarray, batch_size: int = 4096, sample_cap: int = 1000000) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Stream pairwise similarity/distance statistics and collect a bounded
        random sample of (score, label) pairs for classification threshold tuning.

        Returns (distance_metrics, classification_metrics)
        """
        device = self.device

        X = torch.tensor(embeddings, dtype=torch.float32, device=device)
        
        # Ensure labels are numeric and convert to tensor
        if isinstance(labels, np.ndarray):
            if labels.dtype.kind in ['U', 'S']:  # Unicode or byte string
                # Convert string labels to numeric IDs
                unique_labels = np.unique(labels)
                label_to_id = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_id[label] for label in labels])
            else:
                numeric_labels = labels.astype(np.int64)
        elif isinstance(labels, list):
            if all(isinstance(l, str) for l in labels):
                # Convert string labels to numeric IDs
                unique_labels = list(set(labels))
                label_to_id = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_id[label] for label in labels])
            else:
                numeric_labels = np.array(labels, dtype=np.int64)
        else:
            numeric_labels = np.array(labels, dtype=np.int64)
            
        y = torch.tensor(numeric_labels, dtype=torch.long, device=device)
        X_norm = torch.nn.functional.normalize(X, dim=1)
        n = X.shape[0]

        # Reservoir for sampled (score, is_same) entries to perform threshold search
        reservoir_scores = []  # list of floats
        reservoir_labels = []  # list of ints (0/1)
        reserve_k = int(sample_cap)
        seen = 0

        # Welford-style online mean for distances/sims for same/different pairs
        stats = {
            'same': {'count': 0, 'mean_sim': 0.0, 'mean_dist': 0.0},
            'diff': {'count': 0, 'mean_sim': 0.0, 'mean_dist': 0.0}
        }

        for i in range(0, n, batch_size):
            xi = X[i:i+batch_size]
            xi_norm = X_norm[i:i+batch_size]
            yi = y[i:i+batch_size]

            # Compare xi against all subsequent blocks to avoid duplicate pairs
            for j in range(i+1, n, batch_size):
                xj = X[j:j+batch_size]
                xj_norm = X_norm[j:j+batch_size]
                yj = y[j:j+batch_size]

                sims = torch.mm(xi_norm, xj_norm.T)  # [Bi, Bj]
                dists = torch.cdist(xi, xj, p=2)

                # flatten and move to cpu for sampling/stats accumulation
                sims_f = sims.flatten().cpu().numpy()
                dists_f = dists.flatten().cpu().numpy()

                # labels equality
                eq = (yi[:, None] == yj[None, :]).flatten().cpu().numpy()

                # Update online means
                for s_val, d_val, is_same in zip(sims_f, dists_f, eq):
                    key = 'same' if is_same else 'diff'
                    st = stats[key]
                    st['count'] += 1
                    # update running mean for sim
                    delta_sim = s_val - st['mean_sim']
                    st['mean_sim'] += delta_sim / st['count']
                    # update running mean for dist
                    delta_dist = d_val - st['mean_dist']
                    st['mean_dist'] += delta_dist / st['count']

                # Reservoir sampling for classification threshold search
                for s_val, is_same in zip(sims_f, eq):
                    seen += 1
                    if len(reservoir_scores) < reserve_k:
                        reservoir_scores.append(float(s_val))
                        reservoir_labels.append(int(is_same))
                    else:
                        # replace with decreasing probability
                        import random
                        r = random.randrange(seen)
                        if r < reserve_k:
                            reservoir_scores[r] = float(s_val)
                            reservoir_labels[r] = int(is_same)

        # Build numpy arrays for threshold search
        if reservoir_scores:
            import numpy as _np
            scores_np = _np.array(reservoir_scores)
            labels_np = _np.array(reservoir_labels)
        else:
            scores_np = _np.array([])
            labels_np = _np.array([])

        # Derive classification metrics via threshold search on the sampled pairs
        if scores_np.size > 0 and labels_np.size > 0 and labels_np.sum() > 0:
            best_thresh, best_f1 = self._find_optimal_threshold(labels_np, scores_np)
            y_pred = (scores_np > best_thresh).astype(int)
            from sklearn.metrics import (accuracy_score,
                                         precision_recall_fscore_support,
                                         roc_auc_score)
            acc = float(accuracy_score(labels_np, y_pred))
            prec, rec, f1, _ = precision_recall_fscore_support(labels_np, y_pred, average='binary', zero_division=0)
            try:
                auc = float(roc_auc_score(labels_np, scores_np))
            except Exception:
                auc = 0.0
        else:
            best_thresh, best_f1 = float(self.threshold), 0.0
            acc = prec = rec = f1 = auc = 0.0

        distance_metrics = {
            "avg_distance_same_player": float(stats['same']['mean_dist']) if stats['same']['count'] > 0 else float('nan'),
            "avg_distance_different_player": float(stats['diff']['mean_dist']) if stats['diff']['count'] > 0 else float('nan'),
            "avg_similarity_same_player": float(stats['same']['mean_sim']) if stats['same']['count'] > 0 else float('nan'),
            "avg_similarity_different_player": float(stats['diff']['mean_sim']) if stats['diff']['count'] > 0 else float('nan'),
            "distance_separation": float(stats['diff']['mean_dist'] - stats['same']['mean_dist']) if stats['same']['count'] > 0 and stats['diff']['count'] > 0 else float('nan'),
            "similarity_separation": float(stats['same']['mean_sim'] - stats['diff']['mean_sim']) if stats['same']['count'] > 0 and stats['diff']['count'] > 0 else float('nan'),
            "same_player_pairs_count": int(stats['same']['count']),
            "different_player_pairs_count": int(stats['diff']['count']),
        }

        classification_metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "threshold_used": float(best_thresh),
        }

        return distance_metrics, classification_metrics


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
    
    def _evaluate_ranking(self, sims_matrix, labels: np.ndarray, batch_size: int = 4096, use_batched: bool = True, chunk_size: int = 1024) -> Dict[str, float]:
        """
        Compute mean Average Precision (mAP) for retrieval using a precomputed similarity matrix.

        Args:
            sims_matrix: torch.Tensor or numpy array of shape [N, N] with pairwise similarities
            labels: numpy array of labels
        """
        device = self.device

        # Accept either a full square similarity matrix [N,N] or raw embeddings [N,D].
        # If embeddings are provided, compute the full similarity matrix once.
        if not isinstance(sims_matrix, torch.Tensor):
            arr = np.asarray(sims_matrix)
            if arr.ndim != 2:
                raise ValueError("sims_matrix/embeddings must be a 2D array")

            # Square matrix -> treat as similarity matrix; otherwise treat as embeddings
            if arr.shape[0] == arr.shape[1]:
                sims = torch.tensor(arr, dtype=torch.float32, device=device)
            else:
                X = torch.tensor(arr, dtype=torch.float32, device=device)
                X_norm = torch.nn.functional.normalize(X, dim=1)
                sims = torch.mm(X_norm, X_norm.T)
        else:
            t = sims_matrix
            if t.dim() == 2 and t.size(0) == t.size(1):
                sims = t.to(device)
            else:
                X = t.to(device)
                X_norm = torch.nn.functional.normalize(X, dim=1)
                sims = torch.mm(X_norm, X_norm.T)

        # Ensure labels are numeric and convert to tensor
        if isinstance(labels, np.ndarray):
            if labels.dtype.kind in ['U', 'S']:  # Unicode or byte string
                # Convert string labels to numeric IDs
                unique_labels = np.unique(labels)
                label_to_id = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_id[label] for label in labels])
            else:
                numeric_labels = labels.astype(np.int64)
        elif isinstance(labels, list):
            if all(isinstance(l, str) for l in labels):
                # Convert string labels to numeric IDs
                unique_labels = list(set(labels))
                label_to_id = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array([label_to_id[label] for label in labels])
            else:
                numeric_labels = np.array(labels, dtype=np.int64)
        else:
            numeric_labels = np.array(labels, dtype=np.int64)
            
        y = torch.tensor(numeric_labels, dtype=torch.long, device=device)

        n = sims.shape[0]
        ap_list = []

        if not use_batched:
            for i in range(n):
                row = sims[i].clone()
                row[i] = -1e9  # mask self

                # Sort by similarity
                sorted_idx = torch.argsort(row, descending=True)
                sorted_labels = y[sorted_idx].cpu().numpy()

                true_label = y[i].cpu().item()
                relevant = (sorted_labels == true_label).astype(int)

                ap = self._compute_average_precision(relevant)
                ap_list.append(ap)
        else:
            # Batched per-anchor computation: avoid storing NxN. Process anchors in chunks.
            device = self.device
            # If sims is actually an embeddings matrix (non-square), compute X_norm once
            if sims.dim() == 2 and sims.size(0) != sims.size(1):
                X = sims
                X_norm = torch.nn.functional.normalize(X.to(device), dim=1)
            else:
                # sims is square similarity matrix but we still batch to reduce memory peaks
                # We'll compute each anchor's similarity by reading the row in chunks
                X_norm = None

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                for i in range(start, end):
                    if X_norm is not None:
                        row = torch.mm(X_norm[i:i+1], X_norm.T).squeeze(0)
                    else:
                        row = sims[i].clone()

                    row[i] = -1e9
                    sorted_idx = torch.argsort(row, descending=True)
                    sorted_labels = y[sorted_idx].cpu().numpy()

                    true_label = y[i].cpu().item()
                    relevant = (sorted_labels == true_label).astype(int)

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


    def compute_recall_at_k(self, embeddings: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None, ks: Tuple[int, ...] = (1,5,10), sims_matrix: Optional[Any] = None, use_batched: bool = False, chunk_size: int = 1024) -> Dict[str, float]:
        """
        Compute recall@k for the provided embeddings and labels.

        Args:
            embeddings: numpy array of shape [N, D]
            labels: numpy array of shape [N]
            ks: tuple of k values to compute

        Returns:
            Dict with keys 'rank_{k}_accuracy' for each k
        """
        # If a precomputed similarity matrix is provided, use it; otherwise build from embeddings
        device = self.device

        if sims_matrix is None:
            if embeddings is None or labels is None:
                raise ValueError("Either sims_matrix or embeddings+labels must be provided")

            if embeddings.shape[0] == 0:
                return {f'rank_{k}_accuracy': 0.0 for k in ks}

            X = torch.tensor(embeddings, dtype=torch.float32, device=device)
            X_norm = torch.nn.functional.normalize(X, dim=1)
            sims = torch.mm(X_norm, X_norm.T)
            y = np.array(labels)
        else:
            # accept numpy or torch similarity matrix
            if isinstance(sims_matrix, torch.Tensor):
                sims = sims_matrix.to(device)
            else:
                sims = torch.tensor(sims_matrix, dtype=torch.float32, device=device)
            y = np.array(labels)


        n = sims.shape[0]
        recalls = {k: 0 for k in ks}

        if not use_batched:
            for i in range(n):
                row = sims[i].clone()
                row[i] = -1e9

                kmax = max(ks)
                k = min(kmax, n)
                topk = torch.topk(row, k=k).indices.cpu().numpy()
                ranked = y[topk]

                for kk in ks:
                    kk_use = min(kk, k)
                    if y[i] in ranked[:kk_use]:
                        recalls[kk] += 1
        else:
            # Batched mode: compute similarities per-anchor in chunks to avoid NxN memory
            device = self.device
            # If sims is embedding matrix, use normalized embeddings
            if sims.dim() == 2 and sims.size(0) != sims.size(1):
                X_norm = torch.nn.functional.normalize(sims.to(device), dim=1)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    anchors = X_norm[start:end]  # [B, D]
                    sims_chunk = torch.mm(anchors, X_norm.T)  # [B, N]
                    sims_chunk[:, start + torch.arange(end - start)] = -1e9
                    sims_chunk = sims_chunk.cpu()

                    for bi in range(end - start):
                        row = sims_chunk[bi]
                        kmax = max(ks)
                        k = min(kmax, n)
                        topk = torch.topk(row, k=k).indices.numpy()
                        ranked = y[topk]
                        idx = start + bi
                        for kk in ks:
                            kk_use = min(kk, k)
                            if y[idx] in ranked[:kk_use]:
                                recalls[kk] += 1
            else:
                # sims is a precomputed square matrix; iterate anchors in chunks
                sims_cpu = sims.cpu()
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    chunk = sims_cpu[start:end]  # [B, N]
                    for bi in range(end - start):
                        row = chunk[bi].clone()
                        row[start + bi] = -1e9
                        kmax = max(ks)
                        k = min(kmax, n)
                        topk = torch.topk(row, k=k).indices.numpy()
                        ranked = y[topk]
                        idx = start + bi
                        for kk in ks:
                            kk_use = min(kk, k)
                            if y[idx] in ranked[:kk_use]:
                                recalls[kk] += 1

        denom = float(n) if n > 0 else 1.0
        return {f'rank_{k}_accuracy': recalls[k] / denom for k in ks}


    

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
    
    def cleanup(self):
        """
        Clean up evaluation resources to free memory.
        
        Call this method after evaluation is complete to prevent memory leaks.
        """
        logger.info("Cleaning up evaluation resources")
        
        # Clear model reference
        if hasattr(self, 'model'):
            # Move model to CPU to free GPU memory
            if torch.cuda.is_available() and str(self.model.device).startswith('cuda'):
                self.model = self.model.cpu()
                torch.cuda.empty_cache()
            self.model = None
        
        # Clear any cached embeddings or large data structures
        if hasattr(self, '_embedding_cache'):
            self._embedding_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Evaluation cleanup completed")


# Training Progress Metrics Functions
# These functions calculate metrics to monitor training progress and diagnose convergence issues

def calculate_embedding_variance(embeddings: torch.Tensor) -> float:
    """Calculate variance of embedding vectors to measure feature diversity.

    Higher variance indicates more diverse and discriminative features.
    Low variance might indicate collapsed representations or poor feature learning.
    """
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    return torch.var(embeddings, dim=0).mean().item()


def calculate_intra_inter_distances(anchor_emb: torch.Tensor, positive_emb: torch.Tensor,
                                  negative_emb: torch.Tensor) -> Dict[str, float]:
    """Calculate intra-class and inter-class distances for triplet analysis.

    Returns:
        - intra_class_distance: Average distance between anchor and positive (same class)
        - inter_class_distance: Average distance between anchor and negative (different class)
        - triplet_margin_satisfaction: Ratio of triplets that satisfy the margin constraint
    """
    # Intra-class distances (should be small)
    intra_dist = torch.mean(torch.norm(anchor_emb - positive_emb, dim=1)).item()

    # Inter-class distances (should be large)
    inter_dist = torch.mean(torch.norm(anchor_emb - negative_emb, dim=1)).item()

    # Triplet margin satisfaction ratio
    triplet_distances = torch.norm(anchor_emb - positive_emb, dim=1) - torch.norm(anchor_emb - negative_emb, dim=1)
    margin_satisfaction = torch.mean((triplet_distances < 0).float()).item()

    return {
        'intra_class_distance': intra_dist,
        'inter_class_distance': inter_dist,
        'triplet_margin_satisfaction': margin_satisfaction
    }


def calculate_triplet_mining_efficiency(anchor_emb: torch.Tensor, positive_emb: torch.Tensor,
                                       negative_emb: torch.Tensor, margin: float) -> Dict[str, float]:
    """Calculate triplet mining efficiency metrics.

    Analyzes the quality of triplet selection for training:
    - Hard triplets: Challenging cases where positive and negative are close
    - Easy triplets: Trivial cases where positive is much closer than negative
    - Semi-hard triplets: Balanced cases that provide good learning signal

    Returns:
        - hard_triplets_ratio: Ratio of triplets that are challenging to learn
        - easy_triplets_ratio: Ratio of triplets that are too easy
        - semi_hard_triplets_ratio: Ratio of well-balanced triplets
        - mining_efficiency: Overall efficiency score (higher is better)
    """
    # Calculate distances
    pos_dist = torch.norm(anchor_emb - positive_emb, dim=1)
    neg_dist = torch.norm(anchor_emb - negative_emb, dim=1)

    # Hard triplets: where positive distance is close to negative distance
    hard_triplets = torch.mean((pos_dist > neg_dist - margin).float()).item()

    # Easy triplets: where positive is much closer than negative
    easy_triplets = torch.mean((pos_dist < neg_dist - 2 * margin).float()).item()

    # Semi-hard triplets: where negative is closer than positive but still satisfies margin
    semi_hard = torch.mean(((pos_dist < neg_dist) & (neg_dist < pos_dist + margin)).float()).item()

    return {
        'hard_triplets_ratio': hard_triplets,
        'easy_triplets_ratio': easy_triplets,
        'semi_hard_triplets_ratio': semi_hard,
        'mining_efficiency': 1.0 - easy_triplets  # Higher is better (more challenging triplets)
    }


def calculate_gradient_norm(model: torch.nn.Module) -> float:
    """Calculate the norm of gradients for all model parameters.

    Monitors for gradient explosion (very high values) or vanishing (very low values).
    Typical range: 0.01 - 100, depending on model size and learning rate.
    """
    total_norm = 0.0
    param_count = 0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

    return total_norm ** (1. / 2) if param_count > 0 else 0.0


# Backwards-compatible alias: some tests and callers expect `Evaluator`
Evaluator = ModelEvaluator
