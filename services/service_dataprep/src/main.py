"""
Main FastAPI entry point for LaxAI DataPrep Service.
"""
import os
import sys

# Ensure shared_libs can be imported
sys.path.insert(0, '/app')

from shared_libs.utils.env_secrets import setup_environment_secrets

setup_environment_secrets()

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import API routers
from .v1 import router as dataprep_router

logger = logging.getLogger(__name__)


class Settings:
    """Application settings with environment variable support."""

    def __init__(self):
        self.app_name: str = "LaxAI DataPrep Service"
        self.app_version: str = "1.0.0"
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8081"))  # Different port from service_api
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
    logger.info("Starting LaxAI DataPrep Service...")
    yield
    logger.info("Shutting down LaxAI DataPrep Service...")


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
    app.include_router(dataprep_router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "dataprep"}

    return app


# Create the FastAPI app instance
app = create_application()