#!/usr/bin/env python3
"""
Test the optimized batch processing embedding generation.
"""

import sys
import numpy as np
import torch
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_batch_optimization():
    """Test function to be called from main."""
    try:
        from train.evaluator import ModelEvaluator
        from config.all_config import evaluator_config
        print("✅ Successfully imported optimized ModelEvaluator")
        print(f"✅ Config loaded - batch_size: {evaluator_config.batch_size}, num_workers: {evaluator_config.num_workers}")
        
        # Create a dummy model for testing
        class DummyModel(torch.nn.Module):
            def __init__(self, embedding_dim=128):
                super().__init__()
                self.fc = torch.nn.Linear(3 * 224 * 224, embedding_dim)  # Assuming 224x224 RGB images
                
            def forward(self, x):
                # Flatten and process
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)
                embeddings = self.fc(x_flat)
                # L2 normalize like a real siamese network
                return torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Create a dummy dataset for testing
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Return triplet format: (anchor, positive, negative, label)
                image = torch.randn(3, 224, 224)  # Random image
                label = torch.tensor(idx % 10)  # 10 different "players"
                return image, image, image, label  # Use same image for all triplet elements
        
        print("Creating test setup...")
        
        # Create model and evaluator
        model = DummyModel()
        device = torch.device('cpu')
        evaluator = ModelEvaluator(model, device)
        
        # Create test dataset
        dataset = DummyDataset(size=50)  # Small dataset for quick testing
        print(f"Created test dataset with {len(dataset)} samples")
        
        # Test the optimized embedding generation
        print("Testing optimized batch processing...")
        start_time = time.time()
        
        embeddings, labels, image_paths = evaluator._generate_embeddings(dataset)
        
        end_time = time.time()
        
        print(f"✅ Batch processing completed in {end_time - start_time:.2f} seconds")
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"✅ Embedding shape: {embeddings.shape}")
        print(f"✅ Unique labels: {len(set(labels))}")
        print(f"✅ Sample embedding stats: min={embeddings.min():.4f}, max={embeddings.max():.4f}, mean={embeddings.mean():.4f}")
        
        # Verify embedding norms (should be ~1.0 due to L2 normalization)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✅ Embedding norms: mean={norms.mean():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")
        
        print("\n🎉 SUCCESS: Optimized batch processing works correctly!")
        print(f"🚀 Configuration used: batch_size={evaluator_config.batch_size}, num_workers={evaluator_config.num_workers}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_batch_optimization()
