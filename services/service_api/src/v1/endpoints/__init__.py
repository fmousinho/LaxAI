"""API endpoints package."""

from .train import router as train_router
from .track import router as track_router

__all__ = ["train_router", "track_router"]
