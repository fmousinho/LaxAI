"""API endpoints package."""

from v1.endpoints.train import router as train_router
from v1.endpoints.track import router as track_router

__all__ = ["train_router", "track_router"]
