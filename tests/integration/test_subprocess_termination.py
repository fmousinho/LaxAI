import os
import subprocess
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


def find_checkpoint_subprocess(target_checkpoint_path):
    """Find our specific checkpoint upload subprocess by looking for the exact command."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline_str = ' '.join(proc.info['cmdline'])
                if ('wandb_logger.py --upload-checkpoint' in cmdline_str and 
                    target_checkpoint_path in cmdline_str):
                    return proc.info
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


@pytest.mark.slow
def test_checkpoint_subprocess_terminates(tmp_path: Path):
    """
    Test that verifies our checkpoint upload subprocess terminates properly.
    
    This test specifically looks for our checkpoint subprocess and verifies it disappears.
    """
    
    # Initialize wandb logger and create test run
    logger = WandbLogger(enabled=True)
    run_name = f"subprocess_termination_test_{uuid.uuid4().hex[:8]}"
    
    try:
        # Initialize wandb run
        config = {"test_type": "subprocess_termination"}
        logger.init_run(config=config, run_name=run_name, tags=["test", "subprocess_termination"])
        
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "termination_test_checkpoint.pth"
        torch.save({"epoch": 1, "test_data": "termination_test"}, checkpoint_path)
        checkpoint_path_str = str(checkpoint_path)
        
        print(f"📄 Created test checkpoint: {checkpoint_path_str}")
        
        # Verify no checkpoint subprocess exists initially
        initial_subprocess = find_checkpoint_subprocess(checkpoint_path_str)
        assert initial_subprocess is None, f"Checkpoint subprocess found before starting: {initial_subprocess}"
        print("✅ No checkpoint subprocess found initially")
        
        # Trigger checkpoint upload subprocess
        checkpoint_name = logger.get_checkpoint_name()
        print(f"🚀 Starting checkpoint upload subprocess...")
        
        # Use the internal method to trigger upload, but don't wait for completion yet
        logger._trigger_checkpoint_upload_and_cleanup(
            checkpoint_path=checkpoint_path_str,
            checkpoint_name=checkpoint_name,
            epoch=1,
            loss=0.1
        )
        
        # Give the subprocess a moment to start
        time.sleep(1)
        
        # Look for our subprocess while it's running
        running_subprocess = find_checkpoint_subprocess(checkpoint_path_str)
        if running_subprocess:
            print(f"📊 Found running checkpoint subprocess: PID {running_subprocess['pid']}")
        else:
            # It might have already completed - this is also fine
            print("⚡ Checkpoint subprocess completed very quickly")
        
        # Wait for async operations to complete
        print("⏳ Waiting for subprocess to complete...")
        logger._wait_for_pending_operations()
        
        # Give processes a moment to fully terminate
        time.sleep(2)
        
        # Verify the subprocess is gone
        final_subprocess = find_checkpoint_subprocess(checkpoint_path_str)
        assert final_subprocess is None, f"Checkpoint subprocess still running: {final_subprocess}"
        print("✅ Checkpoint subprocess successfully terminated")
        
        # Additional verification: if we found a running subprocess earlier, verify it's the same one that's now gone
        if running_subprocess:
            try:
                proc = psutil.Process(running_subprocess['pid'])
                if proc.is_running():
                    pytest.fail(f"Subprocess PID {running_subprocess['pid']} is still running!")
            except psutil.NoSuchProcess:
                print(f"✅ Confirmed: Subprocess PID {running_subprocess['pid']} has terminated")
            
    finally:
        # Clean up the wandb run
        if logger.run:
            logger.finish()


if __name__ == "__main__":
    # Run the test standalone for debugging
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_checkpoint_subprocess_terminates(Path(tmp_dir))
