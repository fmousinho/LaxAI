"""
Memory Management Guide for Model Evaluation

This module provides best practices and utilities for managing memory
during model evaluation after training.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Any
import torch

logger = logging.getLogger(__name__)

@contextmanager
def evaluation_memory_context(training_instance, evaluator_instance):
    """
    Context manager for safe model evaluation with automatic cleanup.

    Args:
        training_instance: The Training instance with the trained model
        evaluator_instance: The ModelEvaluator instance for evaluation

    Usage:
        with evaluation_memory_context(trainer, evaluator):
            results = evaluator.evaluate_comprehensive(dataset)
    """
    try:
        # Prepare model for evaluation
        eval_model = training_instance.get_model_for_evaluation()
        logger.info("Model prepared for evaluation")

        yield eval_model

    finally:
        # Cleanup after evaluation
        if evaluator_instance:
            evaluator_instance.cleanup()

        # Optional: Cleanup training instance if evaluation is complete
        # training_instance.cleanup_model()  # Uncomment if you want to free training model

        logger.info("Evaluation memory context cleanup completed")

def evaluate_with_memory_management(training_instance, dataset, **eval_kwargs):
    """
    Evaluate a trained model with comprehensive memory management.

    Args:
        training_instance: The Training instance with the trained model
        dataset: Dataset for evaluation
        **eval_kwargs: Additional evaluation arguments

    Returns:
        Evaluation results dictionary
    """
    from train.evaluator import ModelEvaluator

    logger.info("Starting evaluation with memory management")

    # Get model for evaluation
    eval_model = training_instance.get_model_for_evaluation()

    # Create evaluator
    evaluator = ModelEvaluator(
        model=eval_model,
        device=training_instance.device
    )

    try:
        # Run evaluation
        with evaluation_memory_context(training_instance, evaluator):
            results = evaluator.evaluate_comprehensive(dataset, **eval_kwargs)

        logger.info("Evaluation completed successfully")
        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        # Ensure cleanup even on failure
        evaluator.cleanup()
        raise

def get_memory_efficient_evaluation_config():
    """
    Get configuration recommendations for memory-efficient evaluation.

    Returns:
        Dictionary with evaluation configuration recommendations
    """
    return {
        "batch_size": {
            "recommended": 32,
            "memory_constrained": 16,
            "high_memory": 64
        },
        "pairwise_batch_size": {
            "recommended": 4096,
            "memory_constrained": 2048,
            "high_memory": 8192
        },
        "batched_ranking_threshold": {
            "recommended": 5000,
            "memory_constrained": 2000,
            "high_memory": 10000
        },
        "pairwise_sample_cap": {
            "recommended": 1000000,
            "memory_constrained": 500000,
            "high_memory": 2000000
        },
        "memory_management_tips": [
            "Use batched evaluation for large datasets (>5000 samples)",
            "Reduce batch sizes if experiencing memory issues",
            "Call evaluator.cleanup() after evaluation",
            "Consider moving model to CPU after training if GPU memory is limited",
            "Use evaluation_memory_context() for automatic cleanup"
        ]
    }

def log_evaluation_memory_usage(stage: str, evaluator: Optional[Any] = None):
    """
    Log memory usage during evaluation stages.

    Args:
        stage: Description of the current evaluation stage
        evaluator: Optional ModelEvaluator instance for additional logging
    """
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        log_msg = f"Evaluation Memory ({stage}): {memory_mb:.1f}MB RSS"

        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            log_msg += f", {gpu_mb:.1f}MB GPU"

        if evaluator and hasattr(evaluator, 'model') and evaluator.model:
            log_msg += f", Model on {evaluator.model.device}"

        logger.info(log_msg)

    except ImportError:
        logger.debug(f"Evaluation Memory ({stage}) - psutil not available")
    except Exception as e:
        logger.warning(f"Failed to log evaluation memory: {e}")

# Example usage patterns
EVALUATION_MEMORY_EXAMPLES = """
# Example 1: Basic evaluation with cleanup
trainer = Training(...)
trainer.train_and_save(model_class, dataset, model_name)

# Evaluate with memory management
results = evaluate_with_memory_management(trainer, val_dataset)

# Manual cleanup
trainer.cleanup_model()

# Example 2: Using context manager for automatic cleanup
trainer = Training(...)
trainer.train_and_save(model_class, dataset, model_name)

evaluator = ModelEvaluator(trainer.get_model_for_evaluation(), trainer.device)

with evaluation_memory_context(trainer, evaluator):
    results = evaluator.evaluate_comprehensive(val_dataset)

# Memory is automatically cleaned up

# Example 3: Memory-constrained evaluation
config = get_memory_efficient_evaluation_config()
small_batch_evaluator = ModelEvaluator(
    trainer.get_model_for_evaluation(),
    trainer.device
)

# Use smaller batch sizes for memory-constrained environments
results = small_batch_evaluator.evaluate_comprehensive(
    val_dataset,
    batch_size=config['batch_size']['memory_constrained'],
    pairwise_batch_size=config['pairwise_batch_size']['memory_constrained']
)

small_batch_evaluator.cleanup()
"""
