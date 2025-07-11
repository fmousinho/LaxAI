def log_progress(logger, desc, current, total, step=1):
    """Log progress only every 5% or at completion to reduce log verbosity."""
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
