"""Version 1 API endpoints."""

from fastapi import APIRouter

from .endpoints.train import router as train_router
from .endpoints.track import router as track_router
from .endpoints.stitch import router as stitch_router

# Use normalized routing: /api/v1/{service}/...
router = APIRouter()
router.include_router(train_router)
router.include_router(track_router)
router.include_router(stitch_router)

__all__ = ["router"]
