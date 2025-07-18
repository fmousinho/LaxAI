#!/usr/bin/env python3
"""
Example demonstrating checkpoint and resume functionality in pipelines.

This example shows how to:
1. Run a pipeline that can be interrupted
2. Resume from where it left off using checkpoints
3. Understand the checkpoint system behavior
"""

import logging
import sys
import os
from typing import Dict, Any

# Add the project root to the path so we can import from core
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import logging_config
from core.train.dataprep_pipeline import DataPrepPipeline
from config.all_config import DetectionConfig

logger = logging.getLogger(__name__)


def main():
    """
    Demonstrate checkpoint and resume functionality.
    """
    print("=== Pipeline Checkpoint and Resume Example ===")
    print()
    
    # Initialize configuration
    config = DetectionConfig()
    tenant_id = "example_tenant"
    
    try:
        # Create a data preparation pipeline
        pipeline = DataPrepPipeline(
            config=config,
            tenant_id=tenant_id,
            verbose=True,
            save_intermediate=True
        )
        
        print("Pipeline created successfully!")
        print(f"Pipeline name: {pipeline.pipeline_name}")
        print(f"Run GUID: {pipeline.run_guid}")
        print(f"Run folder: {pipeline.run_folder}")
        print()
        
        # Example video path (replace with actual video path)
        video_path = "gs://your-bucket/raw/example_video.mp4"
        
        print("=== First Run (Fresh Start) ===")
        print("Running pipeline without checkpoint resume...")
        
        # Run pipeline without checkpoint resume
        results = pipeline.run(
            video_path=video_path,
            resume_from_checkpoint=False
        )
        
        print(f"Pipeline completed with status: {results['status']}")
        print(f"Steps completed: {results.get('steps_completed', 0)}")
        print(f"Steps failed: {results.get('steps_failed', 0)}")
        
        if results.get('errors'):
            print("Errors encountered:")
            for error in results['errors']:
                print(f"  - {error}")
        
        print()
        
        print("=== Second Run (Resume from Checkpoint) ===")
        print("Running pipeline with checkpoint resume enabled...")
        
        # Create a new pipeline instance with same tenant_id
        # This simulates restarting the application
        pipeline2 = DataPrepPipeline(
            config=config,
            tenant_id=tenant_id,
            verbose=True,
            save_intermediate=True
        )
        
        # Set the same run_guid to target the same checkpoint
        pipeline2.run_guid = pipeline.run_guid
        pipeline2.run_folder = pipeline.run_folder
        
        # Run with checkpoint resume enabled
        results2 = pipeline2.run(
            video_path=video_path,
            resume_from_checkpoint=True
        )
        
        print(f"Pipeline completed with status: {results2['status']}")
        print(f"Steps completed: {results2.get('steps_completed', 0)}")
        print(f"Steps failed: {results2.get('steps_failed', 0)}")
        print(f"Resumed from checkpoint: {results2.get('resumed_from_checkpoint', False)}")
        
        if results2.get('errors'):
            print("Errors encountered:")
            for error in results2['errors']:
                print(f"  - {error}")
        
        print()
        
        print("=== Pipeline Summary ===")
        summary = results2.get('pipeline_summary', {})
        if summary:
            print(f"Total steps: {summary.get('total_steps', 0)}")
            print(f"Completed steps: {summary.get('completed_steps', 0)}")
            print(f"Failed steps: {summary.get('failed_steps', 0)}")
            print(f"Skipped steps: {summary.get('skipped_steps', 0)}")
        
        print()
        print("=== Checkpoint System Benefits ===")
        print("1. Automatic checkpoint saving after each step")
        print("2. Resume from last completed step on interruption")
        print("3. No duplicate work when resuming")
        print("4. Context preservation across restarts")
        print("5. Automatic cleanup on successful completion")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Error running example: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
