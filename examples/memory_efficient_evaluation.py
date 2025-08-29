#!/usr/bin/env python3
"""
Example: Memory-Efficient Model Evaluation

This script demonstrates how to properly evaluate a trained model
with comprehensive memory management to prevent memory leaks.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.all_config import model_config
from train.training import Training
from train.dataset import LacrossePlayerDataset
from utils.evaluation_memory import (
    evaluate_with_memory_management,
    evaluation_memory_context,
    get_memory_efficient_evaluation_config,
    log_evaluation_memory_usage
)
from utils.cpu_memory import log_comprehensive_memory_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_memory_efficient_evaluation():
    """Example of memory-efficient model evaluation."""

    logger.info("Starting memory-efficient evaluation example")

    # Log initial memory state
    log_evaluation_memory_usage("Initial state")
    log_comprehensive_memory_stats("Before evaluation setup")

    try:
        # 1. Setup training (assuming model is already trained)
        trainer = Training()

        # For this example, we'll assume you have a trained model
        # In practice, you would load from wandb or checkpoint
        logger.info("Note: This example assumes a trained model is available")

        # 2. Create evaluation dataset
        # val_dataset = LacrossePlayerDataset(...)  # Your validation dataset

        # 3. Get memory-efficient evaluation config
        eval_config = get_memory_efficient_evaluation_config()
        logger.info(f"Using evaluation config: {eval_config}")

        # 4. Example 1: Simple evaluation with automatic memory management
        logger.info("Example 1: Simple evaluation with automatic cleanup")
        # results = evaluate_with_memory_management(trainer, val_dataset)

        # 5. Example 2: Manual evaluation with context manager
        logger.info("Example 2: Manual evaluation with context manager")

        # eval_model = trainer.get_model_for_evaluation()
        # from train.evaluator import ModelEvaluator
        # evaluator = ModelEvaluator(eval_model, trainer.device)

        # with evaluation_memory_context(trainer, evaluator):
        #     results = evaluator.evaluate_comprehensive(val_dataset)
        #     log_evaluation_memory_usage("During evaluation", evaluator)

        # 6. Example 3: Memory-constrained evaluation
        logger.info("Example 3: Memory-constrained evaluation settings")
        logger.info("Memory management tips:")
        for tip in eval_config['memory_management_tips']:
            logger.info(f"  - {tip}")

        # 7. Cleanup
        # trainer.cleanup_model()
        log_evaluation_memory_usage("After cleanup")
        log_comprehensive_memory_stats("Final state")

        logger.info("Memory-efficient evaluation example completed")

    except Exception as e:
        logger.error(f"Evaluation example failed: {e}")
        raise

def demonstrate_memory_cleanup():
    """Demonstrate proper memory cleanup procedures."""

    logger.info("Demonstrating memory cleanup procedures")

    # This would be your actual training and evaluation code
    trainer = Training()

    try:
        # Your training code here...
        # trainer.train_and_save(...)

        # After training, model is in memory
        log_evaluation_memory_usage("After training")

        # During evaluation
        # evaluator = ModelEvaluator(trainer.get_model_for_evaluation(), trainer.device)
        # results = evaluator.evaluate_comprehensive(dataset)
        log_evaluation_memory_usage("During evaluation")

        # After evaluation - IMPORTANT: Clean up
        # evaluator.cleanup()
        # trainer.cleanup_model()  # Optional: frees the training model too

        log_evaluation_memory_usage("After cleanup")

    except Exception as e:
        logger.error(f"Memory cleanup demonstration failed: {e}")
        # Ensure cleanup even on failure
        if 'trainer' in locals():
            trainer.cleanup_model()
        raise

if __name__ == "__main__":
    print("Memory-Efficient Model Evaluation Examples")
    print("=" * 50)

    # Run examples
    example_memory_efficient_evaluation()
    print("\n" + "=" * 50)
    demonstrate_memory_cleanup()
