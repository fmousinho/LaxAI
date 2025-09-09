#!/bin/bash
# Setup script for Tracking Service

echo "🎯 Setting up Tracking Service environment..."

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install shared libraries in development mode
pip install -e ../../../shared_libs

echo "✅ Tracking Service setup complete!"
