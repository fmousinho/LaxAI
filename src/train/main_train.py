"""
Main FastAPI entry point for LaxAI.
"""
import os

from utils.env_secrets import setup_environment_secrets
setup_environment_secrets()

import config.logging_config

import multiprocessing as mp
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Fork safety fix - must be at the very beginning
try:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Import API routers

from api.v1.endpoints.train import router as cloud_router

logger = logging.getLogger(__name__)


class Settings:
    """Application settings with environment variable support."""
    
    def __init__(self):
        self.app_name: str = "LaxAI Training API"
        self.app_version: str = "1.0.0"
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.reload: bool = os.getenv("RELOAD", "false").lower() == "true"
        
        # CORS settings
        self.cors_origins: List[str] = self._parse_cors_origins()
        self.cors_credentials: bool = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
        
        # Log level
        self.log_level: str = os.getenv("LOG_LEVEL", "info").lower()
    
    def _parse_cors_origins(self) -> List[str]:
        """Parse CORS origins from environment variable."""
        origins_env = os.getenv("CORS_ORIGINS", "*")
        if origins_env == "*":
            return ["*"]
        return [origin.strip() for origin in origins_env.split(",")]


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 %s starting up...", settings.app_name)
    logger.info("Environment: %s", "development" if settings.debug else "production")
    yield
    logger.info("🛑 %s shutting down...", settings.app_name)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="API for managing LaxAI machine learning training pipelines",
        version=settings.app_version,
        lifespan=lifespan,
        debug=settings.debug
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    
    app.include_router(
        cloud_router,
        prefix="/api/v1",
        tags=["cloud"]
    )

    return app


# Create the FastAPI application
app = create_app()


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc",
        "environment": "development" if settings.debug else "production"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "laxai-training-api",
        "version": settings.app_version
    }


def main():
    """Main entry point for running the server."""
    import uvicorn
    
    try:
        logger.info("Starting %s server...", settings.app_name)
        logger.info("Server will be available at: http://%s:%s", settings.host, settings.port)
        logger.info("API documentation: http://%s:%s/docs", settings.host, settings.port)
        
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level=settings.log_level,
            access_log=True
        )
    except Exception as e:
        logger.error("Failed to start server: %s", e)
        raise


if __name__ == "__main__":
    main()
