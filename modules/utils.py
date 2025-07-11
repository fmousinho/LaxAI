import logging
from typing import Optional, Union

import numpy as np
import torch


def l2_normalize_embedding(embedding: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    L2-normalize an embedding vector to unit length.
    
    This function ensures consistent normalization across the codebase.
    Works with both PyTorch tensors and NumPy arrays.
    
    Note: This helper function complements the normalization done in SiameseNet
    (which uses nn.functional.normalize(embedding, p=2, dim=1) for batch processing).
    
    Args:
        embedding: The embedding vector to normalize. Can be:
            - torch.Tensor of shape (embedding_dim,) or (batch_size, embedding_dim)  
            - np.ndarray of shape (embedding_dim,) or (batch_size, embedding_dim)
    
    Returns:
        Normalized embedding of the same type and shape as input
    """
    if isinstance(embedding, torch.Tensor):
        # For PyTorch tensors
        if embedding.dim() == 1:
            # Single embedding vector: (embedding_dim,)
            return torch.nn.functional.normalize(embedding, p=2, dim=0)
        else:
            # Batch of embeddings: (batch_size, embedding_dim)
            return torch.nn.functional.normalize(embedding, p=2, dim=1)
    else:
        # For NumPy arrays
        if embedding.ndim == 1:
            # Single embedding vector: (embedding_dim,)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return embedding  # Avoid division by zero
            return embedding / norm
        else:
            # Batch of embeddings: (batch_size, embedding_dim)
            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1.0, norms)
            return embedding / norms


def log_progress(
    logger: logging.Logger,
    desc: str,
    current: int,
    total: int,
    step: int = 1
) -> None:
    """
    Log progress only every 5% or at completion to reduce log verbosity.
    
    Args:
        logger: Logger instance to use for output
        desc: Description of the process being logged
        current: Current progress count
        total: Total number of items to process
        step: Step size (currently unused but kept for compatibility)
    """
    if total == 0:
        return
    
    percent = 100 * current / total
    
    # Always log first and last
    if current == 1 or current == total:
        logger.info(f"{desc}: {current}/{total} ({percent:.1f}%)")
        return
    
    # Log every 5% threshold
    prev_percent = 100 * (current - 1) / total
    
    # Check if we've crossed a 5% boundary
    if int(percent // 5) > int(prev_percent // 5):
        logger.info(f"{desc}: {current}/{total} ({percent:.1f}%)")


def validate_bbox(bbox: tuple) -> bool:
    """
    Validate that a bounding box has valid dimensions.
    
    Args:
        bbox: Tuple of (x1, y1, x2, y2) coordinates
        
    Returns:
        True if bbox is valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Default value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default
