#!/usr/bin/env python3
"""
Test script for the comprehensive model evaluation system.
"""

import os
import sys
import torch
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.train.train_pipeline import TrainPipeline
from config.all_config import training_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_evaluation_pipeline():
    """
    Test the complete training and evaluation pipeline.
    """
    logger.info("Starting evaluation pipeline test...")
    
    # Sample dataset path (adjust to your actual dataset)
    dataset_path = "/path/to/your/dataset"  # Update this path
    
    # The dataset should have this structure:
    # dataset_path/
    #   ├── train/
    #   │   ├── player1/
    #   │   ├── player2/
    #   │   └── ...
    #   └── val/
    #       ├── player1/
    #       ├── player2/
    #       └── ...
    
    logger.info("Expected dataset structure:")
    logger.info(f"  {dataset_path}/train/  <- Training data (used for model training)")
    logger.info(f"  {dataset_path}/val/    <- Validation data (used for evaluation)")
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset path does not exist: {dataset_path}")
        logger.info("Please update the dataset_path variable in this script")
        logger.info("Ensure your dataset has train/ and val/ subdirectories")
        return
    
    try:
        # Initialize training pipeline
        pipeline = TrainPipeline(
            tenant_id="tenant1",
            verbose=True,
            save_intermediate=True
        )
        
        # Run complete pipeline with evaluation
        logger.info("Running complete training and evaluation pipeline...")
        
        # Step 1: Create dataset
        context = {"dataset_path": dataset_path}
        context = pipeline._create_dataset(context)
        
        if context.get("status") == "error":
            logger.error(f"Dataset creation failed: {context.get('error')}")
            return
        
        logger.info("✅ Dataset creation completed")
        
        # Step 2: Train model
        context = pipeline._train_model(context)
        
        if context.get("status") == "error":
            logger.error(f"Model training failed: {context.get('error')}")
            return
        
        logger.info("✅ Model training completed")
        
        # Step 3: Evaluate model
        context = pipeline._evaluate_model(context)
        
        if context.get("status") == "error":
            logger.error(f"Model evaluation failed: {context.get('error')}")
            return
        
        logger.info("✅ Model evaluation completed")
        
        # Print evaluation summary
        evaluation_summary = context.get('evaluation_summary', {})
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Accuracy: {evaluation_summary.get('accuracy', 0):.4f}")
        print(f"F1-Score: {evaluation_summary.get('f1_score', 0):.4f}")
        print(f"Rank-1 Accuracy: {evaluation_summary.get('rank_1_accuracy', 0):.4f}")
        print(f"Rank-5 Accuracy: {evaluation_summary.get('rank_5_accuracy', 0):.4f}")
        print(f"Mean Average Precision: {evaluation_summary.get('mean_average_precision', 0):.4f}")
        print("="*50)
        
        logger.info("🎉 Complete pipeline test successful!")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def test_evaluation_only():
    """
    Test just the evaluation component with a dummy model.
    """
    from core.train.evaluator import ModelEvaluator
    from core.train.siamesenet import SiameseNet
    from core.train.dataset import LacrossePlayerDataset
    from config.transforms import get_transforms
    
    logger.info("Testing evaluation component only...")
    
    try:
        # Create a dummy model for testing
        model = SiameseNet(embedding_dim=128)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Create evaluator
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            threshold=0.5,
            k_folds=3  # Smaller for testing
        )
        
        logger.info("✅ Model evaluator created successfully")
        
        # Note: To test with real data, you would need to:
        # 1. Load a real dataset
        # 2. Load a trained model
        # 3. Run evaluation
        
        logger.info("Evaluation component test completed")
        logger.info("To test with real data, provide a trained model and dataset")
        
    except Exception as e:
        logger.error(f"Evaluation component test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    print("Model Evaluation System Test")
    print("Choose test mode:")
    print("1. Test complete pipeline (requires dataset)")
    print("2. Test evaluation component only")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_evaluation_pipeline()
    elif choice == "2":
        test_evaluation_only()
    else:
        print("Invalid choice. Exiting.")
