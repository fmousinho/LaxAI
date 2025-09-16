#!/bin/bash
# Setup script for Cloud Service

echo "☁️ Setting up Cloud Service environment..."

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install shared libraries in development mode
pip install -e ../../../shared_libs

echo "✅ Cloud Service setup complete!"
