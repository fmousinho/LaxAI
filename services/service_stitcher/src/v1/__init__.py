"""Version 1 API endpoints."""

from fastapi import APIRouter
from .endpoints.video_endpoint import router as video_router

# Create main v1 router that includes all endpoints

# Use normalized routing: /api/v1/stitcher/...
router = APIRouter(prefix="/stitcher")
router.include_router(video_router)

__all__ = ["router"]