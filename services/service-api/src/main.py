#!/usr/bin/env python3
"""
API Service - Main Entry Point

Handles REST API endpoints and web interface operations.
"""

import uvicorn
from fastapi import FastAPI
from v1.endpoints.train import router as train_router

app = FastAPI(title="LaxAI API Service", version="1.0.0")

# Include training router
app.include_router(
    train_router,
    prefix="/api/v1",
    tags=["training"]
)

@app.get("/")
async def root():
    return {"message": "🚀 LaxAI API Service is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api"}

def main():
    """Main entry point for API service"""
    print("🚀 Starting LaxAI API Service...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
