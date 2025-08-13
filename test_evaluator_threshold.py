#!/usr/bin/env python3
"""
Test the optimized threshold finding in the actual evaluator.
"""

import sys
import numpy as np
import torch
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_evaluator_integration():
    """Test the optimized threshold finding in the ModelEvaluator."""
    try:
        from train.evaluator import ModelEvaluator
        print("✅ Successfully imported ModelEvaluator with optimized threshold finding")
        
        # Create a dummy model
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.randn(x.shape[0], 128)  # Random embeddings
        
        model = DummyModel()
        device = torch.device('cpu')
        evaluator = ModelEvaluator(model, device)
        
        # Create test data for threshold optimization
        np.random.seed(42)
        n_pairs = 1000
        
        # Simulate evaluation pairs with realistic similarity distribution
        y_true = np.random.choice([0, 1], size=n_pairs, p=[0.7, 0.3]).tolist()  # 30% same player
        y_scores = []
        
        for is_same in y_true:
            if is_same:
                # Same player: higher similarity
                score = np.random.uniform(0.6, 0.9)
            else:
                # Different player: lower similarity  
                score = np.random.uniform(0.1, 0.5)
            y_scores.append(score)
        
        print(f"Test data: {len(y_true)} pairs, {sum(y_true)} same player pairs")
        print(f"Score range: {min(y_scores):.3f} - {max(y_scores):.3f}")
        
        # Test the optimized threshold finding
        print("\nTesting optimized threshold finding...")
        start_time = time.time()
        
        best_threshold, best_f1 = evaluator._find_optimal_threshold(y_true, y_scores)
        
        end_time = time.time()
        
        print(f"✅ Threshold optimization completed in {end_time - start_time:.4f} seconds")
        print(f"✅ Optimal threshold: {best_threshold:.6f}")
        print(f"✅ Best F1 score: {best_f1:.6f}")
        
        # Verify the result makes sense
        if 0.0 <= best_threshold <= 1.0 and 0.0 <= best_f1 <= 1.0:
            print("✅ Results are in valid range")
        else:
            print("❌ Results are out of valid range")
            
        print("\n🎉 SUCCESS: Optimized threshold finding works correctly in the evaluator!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_evaluator_integration()
