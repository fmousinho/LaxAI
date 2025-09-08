#!/bin/bash
# Setup script for Training Service

echo "🧠 Setting up Training Service environment..."

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install shared libraries in development mode
pip install -e ../../../shared_libs

echo "✅ Training Service setup complete!"
