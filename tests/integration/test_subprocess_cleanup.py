import os
import time
import uuid
from pathlib import Path

import psutil
import pytest
import torch

from services.service_training.src.wandb_logger import WandbLogger
from shared_libs.utils.env_secrets import setup_environment_secrets

# Ensure secrets are loaded for wandb communication
setup_environment_secrets()


def count_wandb_processes():
    """Count the number of wandb-related processes currently running."""
    wandb_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Only count checkpoint upload subprocess, not wandb's background processes
            if proc.info['cmdline']:
                cmdline_str = ' '.join(proc.info['cmdline'])
                if 'wandb_logger.py --upload-checkpoint' in cmdline_str:
                    wandb_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return wandb_processes


@pytest.mark.slow
def test_subprocess_cleanup_verification(tmp_path: Path):
    """
    Test that verifies subprocess cleanup after checkpoint upload.
    
    This test:
    1. Counts checkpoint upload processes before upload
    2. Triggers a checkpoint upload subprocess
    3. Waits for completion
    4. Counts checkpoint upload processes after upload
    5. Verifies no checkpoint upload processes remain
    """
    
    # Count initial checkpoint upload processes
    initial_processes = count_wandb_processes()
    initial_count = len(initial_processes)
    print(f"📊 Initial checkpoint upload processes: {initial_count}")
    for proc in initial_processes:
        print(f"  - PID {proc['pid']}: {proc['name']} | {' '.join(proc['cmdline'][:3])}...")
    
    # Initialize wandb logger and create test run
    logger = WandbLogger(enabled=True)
    run_name = f"subprocess_cleanup_test_{uuid.uuid4().hex[:8]}"
    
    try:
        # Initialize wandb run
        config = {"test_type": "subprocess_cleanup"}
        logger.init_run(config=config, run_name=run_name, tags=["test", "subprocess_cleanup"])
        
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "cleanup_test_checkpoint.pth"
        torch.save({"epoch": 1, "test_data": "cleanup_test"}, checkpoint_path)
        
        # Trigger checkpoint upload subprocess
        checkpoint_name = logger.get_checkpoint_name()
        logger._trigger_checkpoint_upload_and_cleanup(
            checkpoint_path=str(checkpoint_path),
            checkpoint_name=checkpoint_name,
            epoch=1,
            loss=0.1
        )
        
        # Wait for async operations to complete
        logger._wait_for_pending_operations()
        print("⏳ Subprocess completed, checking for cleanup...")
        
        # Give processes a moment to fully terminate
        time.sleep(2)
        
        # Count final checkpoint upload processes
        final_processes = count_wandb_processes()
        final_count = len(final_processes)
        print(f"📊 Final checkpoint upload processes: {final_count}")
        for proc in final_processes:
            print(f"  - PID {proc['pid']}: {proc['name']} | {' '.join(proc['cmdline'][:3])}...")
        
        # Calculate difference
        process_diff = final_count - initial_count
        
        # Verify no additional checkpoint upload processes remain
        if process_diff > 0:
            print(f"⚠️ WARNING: {process_diff} additional checkpoint upload processes detected!")
            new_processes = [p for p in final_processes if p not in initial_processes]
            for proc in new_processes:
                print(f"  - NEW PID {proc['pid']}: {proc['name']} | {' '.join(proc['cmdline'])}")
            pytest.fail(f"Subprocess cleanup failed: {process_diff} additional checkpoint upload processes remain")
        else:
            print("✅ Subprocess cleanup successful: No additional checkpoint upload processes remain")
            
    finally:
        # Clean up the wandb run
        if logger.run:
            logger.finish()
            
        # Give wandb time to clean up background processes
        time.sleep(3)
        
        # Final process count (should still be 0 for checkpoint upload processes)
        cleanup_processes = count_wandb_processes()
        cleanup_count = len(cleanup_processes)
        print(f"📊 After cleanup checkpoint upload processes: {cleanup_count}")


if __name__ == "__main__":
    # Run the test standalone for debugging
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_subprocess_cleanup_verification(Path(tmp_dir))
