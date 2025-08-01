#!/usr/bin/env python3
"""
Test script to verify that the crop extraction pipeline is working after fixing the 
Google Cloud Storage prefix bug in list_blobs method.
"""

import sys
import os
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.config.all_config import detection_config
from src.train.dataprep_pipeline import DataPrepPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_crop_extraction():
    """Test the crop extraction pipeline to see if it can now find frame images."""
    
    logger.info("=== Testing Crop Extraction Pipeline Fix ===")
    
    # Initialize the pipeline
    tenant_id = "tenant1"
    pipeline = DataPrepPipeline(
        config=detection_config,
        tenant_id=tenant_id,
        verbose=True,
        save_intermediate=True
    )
    
    # Test with a video file (use one that exists in storage)
    test_video = "game1.mp4"  # You can change this to an actual video file
    
    logger.info(f"Testing crop extraction for video: {test_video}")
    logger.info(f"Using tenant_id: {tenant_id}")
    
    try:
        # Run the pipeline
        results = pipeline.run(video_path=test_video)
        
        logger.info("=== Pipeline Results ===")
        logger.info(f"Status: {results.get('status')}")
        
        if results.get('status') == 'completed':
            logger.info("✅ SUCCESS: Pipeline completed successfully!")
            
            # Show context details
            context = results.get('context', {})
            logger.info(f"Generated datasets folder: {context.get('datasets_folder')}")
            logger.info(f"Frame extraction path: {context.get('frame_extraction_path')}")
            logger.info(f"Crop extraction path: {context.get('crop_extraction_path')}")
            
        else:
            logger.error("❌ FAILED: Pipeline did not complete successfully")
            
            # Show error details
            errors = results.get('errors', [])
            for error in errors:
                logger.error(f"Error: {error}")
                
            # Show step statuses
            step_statuses = results.get('step_statuses', {})
            for step_name, status in step_statuses.items():
                logger.info(f"Step {step_name}: {status}")
                
    except Exception as e:
        logger.error(f"Exception occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crop_extraction()
