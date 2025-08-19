"""
Pipeline step tracking utilities for monitoring multi-step processes.

This module provides classes and enums for tracking the status, timing, and metadata
of individual steps in complex pipelines.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class StepStatus(Enum):
    """Enum for individual step status values."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class PipelineStep:
    """
    Represents a single step in a pipeline with comprehensive tracking.
    
    This class tracks the status, timing, errors, and metadata for individual
    steps in a multi-step process, providing detailed monitoring capabilities.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a pipeline step.
        
        Args:
            name: Unique name for the step
            description: Human-readable description of what the step does
        """
        self.name = name
        self.description = description
        self.status = StepStatus.NOT_STARTED
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.output_path: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def start(self):
        """Mark step as started and record start time."""
        self.status = StepStatus.IN_PROGRESS
        self.start_time = datetime.now()

    def complete(self, output_path: Optional[str] = None, metadata: Dict[str, Any] = {}):
        """
        Mark step as completed successfully.
        
        Args:
            output_path: Optional path to step output file
            metadata: Additional metadata about the step execution
        """
        self.status = StepStatus.COMPLETED
        self.end_time = datetime.now()
        self.output_path = output_path
        if metadata:
            self.metadata.update(metadata)
    
    def error(self, error_message: str):
        """
        Mark step as failed with error message.
        
        Args:
            error_message: Description of the error that occurred
        """
        self.status = StepStatus.ERROR
        self.end_time = datetime.now()
        self.error_message = error_message
    
    def skip(self, reason: str):
        """
        Mark step as skipped with reason.
        
        Args:
            reason: Reason why the step was skipped
        """
        self.status = StepStatus.SKIPPED
        self.end_time = datetime.now()
        self.error_message = reason
    
    @property
    def duration(self) -> Optional[float]:
        """
        Get step duration in seconds.
        
        Returns:
            Duration in seconds if step has started and ended, None otherwise
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if step failed with error."""
        return self.status == StepStatus.ERROR
    
    @property
    def is_skipped(self) -> bool:
        """Check if step was skipped."""
        return self.status == StepStatus.SKIPPED
    
    @property
    def is_running(self) -> bool:
        """Check if step is currently running."""
        return self.status == StepStatus.IN_PROGRESS
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert step to dictionary for serialization.
        
        Returns:
            Dictionary representation of the step
        """
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration,
            "error_message": self.error_message,
            "output_path": self.output_path,
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the step."""
        duration_str = f" ({self.duration:.2f}s)" if self.duration else ""
        return f"{self.name}: {self.status.value}{duration_str}"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"PipelineStep(name='{self.name}', status={self.status.value}, duration={self.duration})"
