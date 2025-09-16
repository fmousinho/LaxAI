"""API v1 schemas package."""

from .training import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    ErrorResponse
)

__all__ = [
    "TrainingRequest",
    "TrainingResponse", 
    "TrainingStatus",
    "ErrorResponse"
]