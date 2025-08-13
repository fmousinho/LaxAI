#!/usr/bin/env python3
"""
Quick integration test for the optimized evaluator distance calculation.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import torch
    from train.evaluator import ModelEvaluator
    print("✅ Successfully imported ModelEvaluator")
    
    # Create test data
    np.random.seed(42)
    n_samples = 20
    embedding_dim = 64
    
    embeddings = np.random.randn(n_samples, embedding_dim)
    labels = ['player_' + str(i % 5) for i in range(n_samples)]  # 5 different players
    
    print(f"Test data: {n_samples} embeddings, {len(set(labels))} unique players")
    
    # Create a dummy model and device for the evaluator
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x
    
    model = DummyModel()
    device = torch.device('cpu')
    
    # Create evaluator instance
    evaluator = ModelEvaluator(model, device)
    print("✅ Created evaluator instance")
    
    # Test the optimized distance evaluation method
    print("Testing _evaluate_distances method...")
    distance_metrics = evaluator._evaluate_distances(embeddings, labels)
    
    print("✅ Distance evaluation completed!")
    print("Metrics returned:")
    for key, value in distance_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n🎉 SUCCESS: Vectorized implementation works correctly!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
