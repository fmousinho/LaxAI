"""
Main FastAPI entry point for LaxAI Stitcher Service.
"""

# Ensure shared_libs can be imported
import sys
import os

# Add workspace root to Python path for local development
# Find the directory containing shared_libs by walking up from current file
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = None
for _ in range(10):  # Prevent infinite loop
    if os.path.exists(os.path.join(current_dir, 'shared_libs')):
        workspace_root = current_dir
        break
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:  # Reached filesystem root
        break
    current_dir = parent_dir

if workspace_root and workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# For Docker, /app is already the root
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from shared_libs.utils.env_secrets import setup_environment_secrets
setup_environment_secrets()

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import API routers
from .v1 import router

import shared_libs.config.logging_config

logger = logging.getLogger(__name__)


class Settings:
    """Application settings with environment variable support."""

    def __init__(self):
        self.app_name: str = "LaxAI Stitcher Service"
        self.app_version: str = "1.0.0"
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8080"))  
        self.reload: bool = os.getenv("RELOAD", "false").lower() == "true"
        self.log_level: str = os.getenv("LOG_LEVEL", "info")

        # CORS settings
        self.cors_origins: List[str] = self._parse_cors_origins()
        self.cors_credentials: bool = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"

    def _parse_cors_origins(self) -> List[str]:
        """Parse CORS origins from environment variable."""
        origins_env = os.getenv("CORS_ORIGINS", "*")
        if origins_env == "*":
            return ["*"]
        return [origin.strip() for origin in origins_env.split(",")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("Starting LaxAI Stitcher Service...")
    
    try:
        yield
    finally:
        logger.info("Shutting down LaxAI Stitcher Service...")
        # Cleanup video sessions on shutdown
        from .v1.endpoints.video_endpoint import video_managers
        if video_managers:
            logger.info(f"Cleaning up {len(video_managers)} active video sessions...")
            video_managers.clear()


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = Settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "stitcher"}

    return app


# Create the FastAPI app instance
app = create_application()

if __name__ == "__main__":
    import uvicorn
    settings = Settings()
    uvicorn.run(
        "services.service_stitcher.src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )