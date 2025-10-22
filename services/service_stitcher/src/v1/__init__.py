"""Version 1 API endpoints."""

from fastapi import APIRouter
from .endpoints.video_endpoint import router as video_router

# Create main v1 router that includes all endpoints
# Don't add prefix here since video_router already has /stitcher/video
router = APIRouter()
router.include_router(video_router)

__all__ = ["router"]