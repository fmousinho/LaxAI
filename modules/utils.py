def log_progress(logger, desc, current, total, step=1):
    if current % step == 0 or current == total:
        percent = 100 * current / total if total else 0
        logger.info(f"{desc}: {current}/{total} ({percent:.1f}%)")
