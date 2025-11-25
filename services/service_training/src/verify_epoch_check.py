
import sys
import logging
from typing import Dict, Any, TypedDict

# Mock classes
class StateDicts(TypedDict):
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    lr_scheduler_state_dict: Dict[str, Any]

# Mock logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wandb_logger")

def save_checkpoint_mock(state_dicts: StateDicts, epoch: int):
    # Logic copied from wandb_logger.py for testing
    required_keys = ['model_state_dict', 'optimizer_state_dict', 'lr_scheduler_state_dict']
    missing_keys = [key for key in required_keys if key not in state_dicts]
    if missing_keys:
        raise ValueError(f"checkpoint_data missing required keys: {missing_keys}")

    # Check epoch consistency
    if 'lr_scheduler_state_dict' in state_dicts:
        scheduler_state = state_dicts['lr_scheduler_state_dict']
        if scheduler_state and 'last_epoch' in scheduler_state:
            scheduler_epoch = scheduler_state['last_epoch']
            if scheduler_epoch != epoch:
                print(f"ERROR: Epoch mismatch: Argument epoch={epoch}, Scheduler epoch={scheduler_epoch}")
            else:
                print("SUCCESS: Epochs match")

# Test cases
print("Test 1: Matching epochs")
state_dicts_match = {
    'model_state_dict': {},
    'optimizer_state_dict': {},
    'lr_scheduler_state_dict': {'last_epoch': 10}
}
save_checkpoint_mock(state_dicts_match, 10)

print("\nTest 2: Mismatching epochs")
state_dicts_mismatch = {
    'model_state_dict': {},
    'optimizer_state_dict': {},
    'lr_scheduler_state_dict': {'last_epoch': 5}
}
save_checkpoint_mock(state_dicts_mismatch, 10)
