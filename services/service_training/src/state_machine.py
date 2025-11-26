"""Training job state machine.

Defines valid states and transitions for training jobs.
"""

from enum import Enum
from typing import Set, Dict


class TrainingJobState(str, Enum):
    """Valid states for a training job."""
    QUEUED = "queued"              # Job created, waiting to start
    RUNNING = "running"             # Training in progress
    COMPLETED = "completed"         # Successfully finished
    FAILED = "failed"               # Failed with error
    CANCELLED = "cancelled"         # User cancelled
    AUTO_SUSPENDED = "auto_suspended"  # Auto-suspended for resume


class StateTransition:
    """Defines valid state transitions."""
    
    VALID_TRANSITIONS: Dict[TrainingJobState, Set[TrainingJobState]] = {
        TrainingJobState.QUEUED: {
            TrainingJobState.RUNNING,
            TrainingJobState.CANCELLED,
            TrainingJobState.FAILED
        },
        TrainingJobState.RUNNING: {
            TrainingJobState.COMPLETED,
            TrainingJobState.FAILED,
            TrainingJobState.CANCELLED,
            TrainingJobState.AUTO_SUSPENDED
        },
        TrainingJobState.AUTO_SUSPENDED: {
            TrainingJobState.QUEUED,  # Re-queued on resume
            TrainingJobState.FAILED,
            TrainingJobState.CANCELLED
        },
        TrainingJobState.COMPLETED: set(),  # Terminal
        TrainingJobState.FAILED: set(),     # Terminal
        TrainingJobState.CANCELLED: set()   # Terminal
    }
    
    @classmethod
    def is_valid_transition(cls, from_state: TrainingJobState, to_state: TrainingJobState) -> bool:
        """Check if transition is valid.
        
        Args:
            from_state: Current state
            to_state: Target state
            
        Returns:
            True if transition is valid
        """
        return to_state in cls.VALID_TRANSITIONS.get(from_state, set())
    
    @classmethod
    def is_terminal_state(cls, state: TrainingJobState) -> bool:
        """Check if state is terminal.
        
        Args:
            state: State to check
            
        Returns:
            True if state is terminal (no valid transitions out)
        """
        return len(cls.VALID_TRANSITIONS.get(state, set())) == 0
