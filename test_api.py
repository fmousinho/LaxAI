#!/usr/bin/env python3
"""
Simple test server for checking API documentation
"""
import os
import sys

# Add paths to sys.path
sys.path.insert(0, '/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/shared_libs')
sys.path.insert(0, '/Users/fernandomousinho/Documents/Learning_to_Code/LaxAI/services/service_training/src')

import uvicorn
from fastapi import FastAPI
from schemas.training import TrainingRequest

app = FastAPI(title="LaxAI Training API Test", version="1.0.0")

@app.post("/train")
async def train(request: TrainingRequest):
    """Training endpoint for testing schema documentation"""
    return {"message": "This is a test endpoint", "request": request.model_dump()}

if __name__ == "__main__":
    print("Starting test API server...")
    print("API docs will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)