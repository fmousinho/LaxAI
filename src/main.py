"""
Main FastAPI entry point for LaxAI.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import API routers
from .api.v1.endpoints.train import router as train_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    """
    logger.info("🚀 LaxAI API starting up...")
    yield
    logger.info("🛑 LaxAI API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="LaxAI Training API",
    description="API for managing LaxAI machine learning training pipelines",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    train_router,
    prefix="/api/v1",
    tags=["training"]
)


@app.get("/")
async def root():
    """
    Root endpoint providing API information.
    """
    return {
        "message": "Welcome to LaxAI Training API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "service": "laxai-training-api"}


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting LaxAI API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
