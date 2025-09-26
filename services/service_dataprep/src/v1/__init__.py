"""Version 1 API endpoints."""

from fastapi import APIRouter
from .endpoints.dataprep import router as dataprep_router

# Create main v1 router that includes all endpoints
router = APIRouter()
router.include_router(dataprep_router)

__all__ = ["router"]