"""Version 1 API endpoints."""

import logging
from fastapi import APIRouter
from v1.endpoints.video_endpoint import router as video_router

logger = logging.getLogger(__name__)

# Create main v1 router that includes all endpoints
# Don't add prefix here since video_router already has /stitcher/video
router = APIRouter()
router.include_router(video_router)
logger.warning("v1 router initialized: video endpoints included under /stitcher/video")

__all__ = ["router"]