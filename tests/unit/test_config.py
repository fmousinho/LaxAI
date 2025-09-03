"""
Test configuration constants for WandB and other tests.
These constants ensure consistent test behavior and performance.
"""

# Test performance constraints
DEFAULT_MAX_EPOCHS = 2  # Maximum epochs for tests unless specifically required
DEFAULT_SINGLE_DATASET = True  # Use single dataset unless test specifically needs more
DEFAULT_MINIMAL_CHECKPOINT_SIZE = True  # Use minimal tensor sizes for checkpoints

# Test timeouts and limits
MAX_TEST_DURATION_SECONDS = 30  # Maximum time for individual WandB tests
MINIMAL_TENSOR_SIZE = (2, 2)  # Minimal tensor dimensions for test checkpoints
MAX_BATCH_SIZE_TEST = 4  # Maximum batch size for test data

# WandB test specific settings
WANDB_TEST_PROJECT_SUFFIX = "_test"  # Suffix for test projects
WANDB_CLEANUP_ENABLED = True  # Always enable cleanup for tests
WANDB_FAST_MODE_DEFAULT = True  # Prefer fast mode for development

def get_test_epochs(specific_requirement: int = None) -> int:
    """
    Get the number of epochs for a test.
    
    Args:
        specific_requirement: If a test specifically needs more epochs, pass the number
        
    Returns:
        Number of epochs to use (defaults to 2 unless specifically required)
    """
    if specific_requirement is not None and specific_requirement > DEFAULT_MAX_EPOCHS:
        # Only allow more epochs if explicitly justified
        print(f"Warning: Test using {specific_requirement} epochs (more than default {DEFAULT_MAX_EPOCHS})")
        return specific_requirement
    return DEFAULT_MAX_EPOCHS

def get_minimal_checkpoint_data(epoch: int = 1):
    """
    Create minimal checkpoint data for testing.
    
    Args:
        epoch: Epoch number for the checkpoint
        
    Returns:
        Dictionary with minimal checkpoint data
    """
    import torch
    
    return {
        "epoch": epoch,
        "model_state_dict": {"test_param": torch.tensor([1.0])},  # Minimal tensor
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.001}]},
        "loss": 0.5 - epoch * 0.1,
        "model_name": f"test_model_epoch_{epoch}"
    }


def enforce_test_limits(func):
    """
    Decorator to enforce test performance limits.
    
    This decorator can be used on test functions to ensure they follow
    the 2-epoch, single-dataset policy.
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Run the test
        result = func(*args, **kwargs)
        
        # Check execution time
        duration = time.time() - start_time
        if duration > MAX_TEST_DURATION_SECONDS:
            print(f"Warning: Test {func.__name__} took {duration:.2f}s (exceeds {MAX_TEST_DURATION_SECONDS}s limit)")
        
        return result
    
    return wrapper


def validate_epoch_usage(epochs_used: int, test_name: str = "unknown"):
    """
    Validate that a test is using an appropriate number of epochs.
    
    Args:
        epochs_used: Number of epochs the test is planning to use
        test_name: Name of the test for warning messages
    """
    if epochs_used > DEFAULT_MAX_EPOCHS:
        print(f"Warning: Test '{test_name}' using {epochs_used} epochs (exceeds recommended {DEFAULT_MAX_EPOCHS})")
        print(f"Consider reducing epochs or using get_test_epochs() for explicit justification")
