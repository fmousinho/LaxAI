import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../services/service_training/src")))

from wandb_logger import _upload_checkpoint_in_subprocess

@patch("wandb_logger.wandb")
@patch("wandb_logger.subprocess.run")
def test_upload_checkpoint_deletes_run(mock_subprocess, mock_wandb):
    """Test that the temporary run is deleted after upload."""
    # Setup mocks
    mock_run = MagicMock()
    mock_run.id = "test_run_id"
    mock_wandb.init.return_value = mock_run
    
    mock_api = MagicMock()
    mock_run_obj = MagicMock()
    mock_api.run.return_value = mock_run_obj
    mock_wandb.Api.return_value = mock_api
    
    # Create a dummy checkpoint file
    checkpoint_path = "test_checkpoint.pth"
    with open(checkpoint_path, "w") as f:
        f.write("dummy data")
        
    try:
        # Call the function
        _upload_checkpoint_in_subprocess(
            run_id="original_run_id",
            checkpoint_path=checkpoint_path,
            checkpoint_name="test_checkpoint",
            project="test_project",
            entity="test_entity"
        )
        
        # Verify run.finish() was called
        mock_run.finish.assert_called_once()
        
        # Verify run.delete() was called with correct args
        mock_wandb.Api.assert_called_once()
        mock_api.run.assert_called_with("test_entity/test_project/test_run_id")
        mock_run_obj.delete.assert_called_once_with(delete_artifacts=False)
        
    finally:
        # Cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

@patch("wandb_logger.wandb")
@patch("wandb_logger.subprocess.run")
def test_upload_checkpoint_handles_delete_error(mock_subprocess, mock_wandb):
    """Test that the process doesn't crash if run deletion fails."""
    # Setup mocks
    mock_run = MagicMock()
    mock_run.id = "test_run_id"
    mock_wandb.init.return_value = mock_run
    
    mock_api = MagicMock()
    mock_run_obj = MagicMock()
    # Simulate delete failure
    mock_run_obj.delete.side_effect = Exception("Delete failed")
    mock_api.run.return_value = mock_run_obj
    mock_wandb.Api.return_value = mock_api
    
    # Create a dummy checkpoint file
    checkpoint_path = "test_checkpoint_error.pth"
    with open(checkpoint_path, "w") as f:
        f.write("dummy data")
        
    try:
        # Call the function - should not raise exception
        _upload_checkpoint_in_subprocess(
            run_id="original_run_id",
            checkpoint_path=checkpoint_path,
            checkpoint_name="test_checkpoint",
            project="test_project",
            entity="test_entity"
        )
        
        # Verify run.finish() was called
        mock_run.finish.assert_called_once()
        
        # Verify attempt to delete was made
        mock_run_obj.delete.assert_called_once()
        
    finally:
        # Cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
