#!/usr/bin/env python3
"""
API Service - Main Entry Point

Handles REST API endpoints and web interface operations.
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="LaxAI API Service", version="1.0.0")

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
