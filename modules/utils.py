import logging
from typing import Optional


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
