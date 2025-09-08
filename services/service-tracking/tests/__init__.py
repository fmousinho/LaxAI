"""
Independent test suite for TrackGeneratorPipeline functionality.

This test suite can be deployed and run independently from the main LaxAI test suite,
containing all necessary mocks and fixtures for testing the TrackGeneratorPipeline
in isolation.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))