#!/usr/bin/env python3
"""
Startup script for the LaxAI Training API.
"""
import sys
import os
import logging

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the FastAPI app
from main import app

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting LaxAI Training API...")
    logger.info("📖 API Documentation available at: http://localhost:8000/docs")
    logger.info("📚 ReDoc available at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
