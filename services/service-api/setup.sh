#!/bin/bash
# Setup script for API Service

echo "🚀 Setting up API Service environment..."

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install shared libraries in development mode
pip install -e ../../../shared_libs

echo "✅ API Service setup complete!"
