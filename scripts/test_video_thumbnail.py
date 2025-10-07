#!/usr/bin/env python3
"""
Test script to run the tracking pipeline on a specific video and generate thumbnail.

This script initializes the TrackGeneratorPipeline and processes the video at
gs://laxai_dev/test_tenant/process/test_video_for_stiching/imported/test_video.mp4
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add service paths to Python path
sys.path.insert(0, str(project_root / "services" / "service_tracking" / "src"))
sys.path.insert(0, str(project_root / "shared_libs"))

def main():
    """Run the tracking pipeline test."""
    try:
        # Import required modules
        from services.service_tracking.src.unverified_track_generator_pipeline import TrackGeneratorPipeline
        from shared_libs.config.all_config import DetectionConfig

        # Initialize configuration
        config = DetectionConfig()

        # Initialize pipeline
        print("Initializing TrackGeneratorPipeline...")
        pipeline = TrackGeneratorPipeline(
            config=config,
            tenant_id="test_tenant",
            verbose=True
        )

        # Run pipeline on the video
        video_path = "test_tenant/process/test_video_for_stiching/imported/test_video.mp4"
        print(f"Running pipeline on video: {video_path}")

        results = pipeline.run(video_path)

        # Print results
        print("\nPipeline Results:")
        print(f"Status: {results['status']}")
        print(f"Video GUID: {results.get('video_guid', 'N/A')}")
        print(f"Video Folder: {results.get('video_folder', 'N/A')}")

        if results['status'] == 'completed':
            print("✅ Pipeline completed successfully!")
            print(f"Run GUID: {results.get('run_guid', 'N/A')}")
            print(f"Pipeline Summary: {results.get('pipeline_summary', {})}")
        else:
            print("❌ Pipeline failed or was cancelled")
            print(f"Errors: {results.get('errors', [])}")

        return results

    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()