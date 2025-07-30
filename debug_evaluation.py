#!/usr/bin/env python3
"""
Debug script to test model evaluation and diagnose zero metrics issue.
"""

import logging
import torch
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_evaluation():
    """Test the model evaluation pipeline with debugging."""
    
    # Import after logging setup
    from core.train.evaluator import ModelEvaluator
    from core.train.siamesenet import SiameseNet
    from core.config.all_config import model_config
    from core.train.wandb_logger import wandb_logger
    
    # Initialize device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Test 1: Check if we can create a model and generate a dummy embedding
    logger.info("=" * 60)
    logger.info("TEST 1: Model initialization and basic forward pass")
    logger.info("=" * 60)
    
    try:
        # Create a fresh model
        model = SiameseNet(embedding_dim=model_config.embedding_dim)
        model.to(device)
        model.eval()
        
        logger.info(f"✓ Model created successfully")
        logger.info(f"  - Embedding dimension: {model.embedding_dim}")
        logger.info(f"  - Model device: {next(model.parameters()).device}")
        logger.info(f"  - Model is in eval mode: {not model.training}")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 120, 80).to(device)  # Batch size 1
        logger.info(f"  - Dummy input shape: {dummy_input.shape}")
        logger.info(f"  - Dummy input stats: min={dummy_input.min():.4f}, max={dummy_input.max():.4f}, mean={dummy_input.mean():.4f}")
        
        with torch.no_grad():
            dummy_embedding = model(dummy_input)
            
        logger.info(f"  - Dummy embedding shape: {dummy_embedding.shape}")
        logger.info(f"  - Dummy embedding stats: min={dummy_embedding.min():.4f}, max={dummy_embedding.max():.4f}, mean={dummy_embedding.mean():.4f}")
        logger.info(f"  - Embedding norm: {torch.norm(dummy_embedding, p=2, dim=1):.4f}")
        
        # Check if embedding is all zeros
        if torch.allclose(dummy_embedding, torch.zeros_like(dummy_embedding), atol=1e-8):
            logger.error("❌ Dummy embedding is all zeros!")
            return False
        else:
            logger.info("✓ Dummy embedding is non-zero")
            
    except Exception as e:
        logger.error(f"❌ Model creation/forward pass failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False
    
    # Test 2: Try to load model from wandb
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Loading model from wandb registry")
    logger.info("=" * 60)
    
    try:
        loaded_model = wandb_logger.load_model_from_registry(
            model_class=lambda: SiameseNet(embedding_dim=model_config.embedding_dim),
            collection_name="PlayerEmbeddings",
            alias="latest",
            device=str(device)
        )
        
        if loaded_model is not None:
            logger.info("✓ Model loaded successfully from wandb")
            logger.info(f"  - Model device: {next(loaded_model.parameters()).device}")
            
            # Test loaded model with dummy input
            loaded_model.eval()
            with torch.no_grad():
                loaded_embedding = loaded_model(dummy_input)
                
            logger.info(f"  - Loaded model embedding shape: {loaded_embedding.shape}")
            logger.info(f"  - Loaded model embedding stats: min={loaded_embedding.min():.4f}, max={loaded_embedding.max():.4f}, mean={loaded_embedding.mean():.4f}")
            logger.info(f"  - Loaded embedding norm: {torch.norm(loaded_embedding, p=2, dim=1):.4f}")
            
            # Check if loaded embedding is all zeros
            if torch.allclose(loaded_embedding, torch.zeros_like(loaded_embedding), atol=1e-8):
                logger.error("❌ Loaded model embedding is all zeros!")
                logger.error("❌ The saved model appears to be untrained or corrupted!")
                model_to_use = model  # Use fresh model instead
                logger.info("🔄 Will use fresh untrained model for evaluation")
            else:
                logger.info("✓ Loaded model embedding is non-zero")
                model_to_use = loaded_model
        else:
            logger.warning("⚠️ Could not load model from wandb, using fresh model")
            model_to_use = model
            
    except Exception as e:
        logger.error(f"❌ Loading model from wandb failed: {e}")
        model_to_use = model
        logger.info("🔄 Will use fresh untrained model for evaluation")
    
    # Test 3: Check dataset path
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Dataset availability check")
    logger.info("=" * 60)
    
    # Look for dataset in common locations
    possible_dataset_paths = [
        "/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/datasets/frame19/train",
        "/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/datasets/frame_dataset/train",
        "/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/data/train",
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if Path(path).exists():
            dataset_path = path
            logger.info(f"✓ Found dataset at: {dataset_path}")
            break
    
    if dataset_path is None:
        logger.error("❌ No dataset found in common locations")
        logger.error("Please ensure you have a dataset with the following structure:")
        logger.error("  dataset_path/train/player1/image1.jpg")
        logger.error("  dataset_path/train/player2/image1.jpg")
        logger.error("  dataset_path/val/player1/image2.jpg")
        logger.error("  etc.")
        return False
    
    # Check validation dataset
    val_path = str(Path(dataset_path).parent / "val")
    if Path(val_path).exists():
        logger.info(f"✓ Found validation dataset at: {val_path}")
    else:
        logger.warning(f"⚠️ No validation dataset found at: {val_path}")
        logger.info("Evaluation will create a random split from training data")
    
    # Test 4: Initialize evaluator and run a quick test
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Model evaluation test")
    logger.info("=" * 60)
    
    try:
        evaluator = ModelEvaluator(
            model=model_to_use,
            device=device,
            threshold=0.5,  # Lower threshold for testing
            k_folds=3       # Fewer folds for faster testing
        )
        
        logger.info("✓ ModelEvaluator initialized successfully")
        
        # Run evaluation
        logger.info("🚀 Starting model evaluation...")
        results = evaluator.evaluate_comprehensive(
            dataset_path=dataset_path,
            storage_client=None,
            use_validation_split=True
        )
        
        logger.info("✅ Evaluation completed successfully!")
        
        # Print key results
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 60)
        
        cls_metrics = results.get('classification_metrics', {})
        rank_metrics = results.get('ranking_metrics', {})
        dist_metrics = results.get('distance_metrics', {})
        
        logger.info(f"Classification Metrics:")
        logger.info(f"  - Accuracy: {cls_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  - Precision: {cls_metrics.get('precision', 0):.4f}")
        logger.info(f"  - Recall: {cls_metrics.get('recall', 0):.4f}")
        logger.info(f"  - F1-Score: {cls_metrics.get('f1_score', 0):.4f}")
        
        logger.info(f"Ranking Metrics:")
        logger.info(f"  - Rank-1 Accuracy: {rank_metrics.get('rank_1_accuracy', 0):.4f}")
        logger.info(f"  - Rank-5 Accuracy: {rank_metrics.get('rank_5_accuracy', 0):.4f}")
        logger.info(f"  - Mean Average Precision: {rank_metrics.get('mean_average_precision', 0):.4f}")
        
        logger.info(f"Distance Metrics:")
        logger.info(f"  - Same player avg distance: {dist_metrics.get('avg_distance_same_player', 'NaN')}")
        logger.info(f"  - Different player avg distance: {dist_metrics.get('avg_distance_different_player', 'NaN')}")
        
        # Check if we still have zero metrics
        all_zeros = all([
            cls_metrics.get('accuracy', 0) == 0,
            cls_metrics.get('precision', 0) == 0,
            cls_metrics.get('recall', 0) == 0,
            cls_metrics.get('f1_score', 0) == 0,
            rank_metrics.get('rank_1_accuracy', 0) == 0,
            rank_metrics.get('rank_5_accuracy', 0) == 0,
            rank_metrics.get('mean_average_precision', 0) == 0
        ])
        
        if all_zeros:
            logger.error("❌ All metrics are still zero - there's a fundamental issue!")
            logger.error("🔍 Possible causes:")
            logger.error("  1. Model is not properly trained (all embeddings are similar)")
            logger.error("  2. Dataset has only one player or insufficient variety")
            logger.error("  3. Image preprocessing issues")
            logger.error("  4. Model architecture problems")
            return False
        else:
            logger.info("✅ Some metrics are non-zero - evaluation is working!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🔍 Starting model evaluation debug test...")
    success = test_model_evaluation()
    
    if success:
        logger.info("✅ Debug test completed successfully!")
    else:
        logger.error("❌ Debug test failed - please check the issues above")
