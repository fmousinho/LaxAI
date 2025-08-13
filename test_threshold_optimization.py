#!/usr/bin/env python3
"""
Test the optimized threshold finding implementation.
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def old_find_optimal_threshold(y_true, y_scores, default_threshold=0.5):
    """Original implementation for comparison."""
    thresholds = np.linspace(min(y_scores), max(y_scores), 50)
    best_f1 = 0.0
    best_threshold = default_threshold
    
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


def new_find_optimal_threshold(y_true, y_scores, default_threshold=0.5):
    """New vectorized implementation."""
    if not y_true or not y_scores:
        return default_threshold, 0.0
        
    # Convert to numpy arrays for vectorized operations
    y_true_arr = np.array(y_true, dtype=np.int32)
    y_scores_arr = np.array(y_scores, dtype=np.float32)
    
    # Use unique scores as candidate thresholds for more efficient search
    unique_scores = np.unique(y_scores_arr)
    
    # If too many unique scores, subsample for performance
    if len(unique_scores) > 200:
        # Use percentiles to get a representative sample
        thresholds = np.percentile(y_scores_arr, np.linspace(1, 99, 100))
    else:
        thresholds = unique_scores
    
    # Remove duplicates and sort
    thresholds = np.unique(thresholds)
    
    # Vectorized prediction calculation for all thresholds at once
    predictions = y_scores_arr[:, np.newaxis] > thresholds[np.newaxis, :]
    
    # Vectorized confusion matrix calculation
    tp = np.sum((y_true_arr[:, np.newaxis] == 1) & predictions, axis=0)
    fp = np.sum((y_true_arr[:, np.newaxis] == 0) & predictions, axis=0)
    fn = np.sum((y_true_arr[:, np.newaxis] == 1) & ~predictions, axis=0)
    
    # Vectorized precision, recall, F1 calculation with safe division
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1_scores = np.divide(2 * precision * recall, precision + recall, 
                         out=np.zeros_like(precision, dtype=float), where=(precision + recall) > 0)
    
    # Find the threshold that maximizes F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    
    # Fallback to original threshold if no improvement found
    if best_f1 == 0.0:
        return default_threshold, 0.0
        
    return best_threshold, best_f1


def test_threshold_optimization():
    """Test both implementations and compare performance."""
    print("Testing threshold optimization implementations...")
    
    # Create test data with realistic distribution
    np.random.seed(42)
    n_samples = 5000  # Larger dataset to see more dramatic speedup
    
    # Generate similarity scores with some structure
    # Same player pairs: higher similarity (0.6-0.9)
    # Different player pairs: lower similarity (0.1-0.5)
    same_player_scores = np.random.uniform(0.6, 0.9, n_samples // 3)
    diff_player_scores = np.random.uniform(0.1, 0.5, 2 * n_samples // 3)
    
    y_scores = np.concatenate([same_player_scores, diff_player_scores])
    y_true = np.concatenate([np.ones(len(same_player_scores)), np.zeros(len(diff_player_scores))])
    
    # Shuffle to mix the data
    indices = np.random.permutation(len(y_scores))
    y_scores = y_scores[indices].tolist()
    y_true = y_true[indices].astype(int).tolist()
    
    print(f"Test data: {len(y_scores)} samples")
    print(f"True labels: {sum(y_true)} same player, {len(y_true) - sum(y_true)} different player")
    print(f"Score range: {min(y_scores):.3f} - {max(y_scores):.3f}")
    
    # Test old implementation
    print("\nTesting original implementation...")
    start_time = time.time()
    old_threshold, old_f1 = old_find_optimal_threshold(y_true, y_scores)
    old_time = time.time() - start_time
    print(f"Original: threshold={old_threshold:.6f}, F1={old_f1:.6f}, time={old_time:.4f}s")
    
    # Test new implementation
    print("\nTesting optimized implementation...")
    start_time = time.time()
    new_threshold, new_f1 = new_find_optimal_threshold(y_true, y_scores)
    new_time = time.time() - start_time
    print(f"Optimized: threshold={new_threshold:.6f}, F1={new_f1:.6f}, time={new_time:.4f}s")
    
    # Compare results
    print(f"\nPerformance comparison:")
    print(f"Speedup: {old_time/new_time:.1f}x faster")
    print(f"Threshold difference: {abs(old_threshold - new_threshold):.6f}")
    print(f"F1 difference: {abs(old_f1 - new_f1):.6f}")
    
    # Verify results are similar (allowing for small differences due to different threshold sampling)
    threshold_close = abs(old_threshold - new_threshold) < 0.1
    f1_close = abs(old_f1 - new_f1) < 0.05
    
    if threshold_close and f1_close:
        print("\n✅ SUCCESS: Results are very similar!")
        print(f"🚀 Optimization provides {old_time/new_time:.1f}x speedup with comparable accuracy")
    else:
        print("\n⚠️  WARNING: Results differ significantly")
        print("This might be due to different threshold sampling strategies")
        
        # Check if new implementation found a better solution
        if new_f1 > old_f1:
            print(f"✅ New implementation found better F1 score: {new_f1:.6f} vs {old_f1:.6f}")
    
    return old_time, new_time, old_f1, new_f1


if __name__ == "__main__":
    test_threshold_optimization()
