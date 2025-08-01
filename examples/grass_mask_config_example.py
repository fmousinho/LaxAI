#!/usr/bin/env python3
"""
Example usage of DataPrepPipeline with grass mask configuration.

This example demonstrates how to configure the DataPrepPipeline to enable or disable
grass mask functionality (background removal from player crops).
"""

import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.config.all_config import DetectionConfig, transform_config
from src.train.dataprep_pipeline import DataPrepPipeline


def example_grass_mask_enabled():
    """Example: Create pipeline with grass mask enabled."""
    print("=== Example: DataPrepPipeline with Grass Mask ENABLED ===")
    
    config = DetectionConfig()
    
    # Create pipeline with grass mask explicitly enabled
    pipeline = DataPrepPipeline(
        config=config,
        tenant_id="example_tenant",
        enable_grass_mask=True,  # Explicitly enable grass mask
        verbose=True
    )
    
    print(f"Grass mask enabled: {pipeline.enable_grass_mask}")
    print(f"Pipeline steps: {list(pipeline.step_definitions.keys())}")
    print(f"Has calculate_grass_mask: {'calculate_grass_mask' in pipeline.step_definitions}")
    print(f"Has remove_crop_background: {'remove_crop_background' in pipeline.step_definitions}")
    print()
    
    # Now you can run the pipeline with grass mask functionality
    # result = pipeline.run("path/to/video.mp4")
    
    return pipeline


def example_grass_mask_disabled():
    """Example: Create pipeline with grass mask disabled."""
    print("=== Example: DataPrepPipeline with Grass Mask DISABLED ===")
    
    config = DetectionConfig()
    
    # Create pipeline with grass mask explicitly disabled
    pipeline = DataPrepPipeline(
        config=config,
        tenant_id="example_tenant",
        enable_grass_mask=False,  # Explicitly disable grass mask
        verbose=True
    )
    
    print(f"Grass mask enabled: {pipeline.enable_grass_mask}")
    print(f"Pipeline steps: {list(pipeline.step_definitions.keys())}")
    print(f"Has calculate_grass_mask: {'calculate_grass_mask' in pipeline.step_definitions}")
    print(f"Has remove_crop_background: {'remove_crop_background' in pipeline.step_definitions}")
    print()
    
    # Now you can run the pipeline without grass mask functionality (faster processing)
    # result = pipeline.run("path/to/video.mp4")
    
    return pipeline


def example_default_config():
    """Example: Create pipeline using default configuration."""
    print("=== Example: DataPrepPipeline with DEFAULT Configuration ===")
    
    config = DetectionConfig()
    
    # Create pipeline without specifying enable_grass_mask
    # This will use the default value from transform_config.enable_background_removal
    pipeline = DataPrepPipeline(
        config=config,
        tenant_id="example_tenant",
        verbose=True
        # enable_grass_mask not specified - uses default from transform_config
    )
    
    print(f"Default transform_config.enable_background_removal: {transform_config.enable_background_removal}")
    print(f"Grass mask enabled: {pipeline.enable_grass_mask}")
    print(f"Pipeline steps: {list(pipeline.step_definitions.keys())}")
    print()
    
    return pipeline


def modify_global_config_example():
    """Example: Modify global config and create pipeline."""
    print("=== Example: Modify Global Config ===")
    
    # You can modify the global transform_config to change the default behavior
    original_setting = transform_config.enable_background_removal
    print(f"Original setting: {original_setting}")
    
    # Change the global setting
    transform_config.enable_background_removal = True
    print(f"Modified setting: {transform_config.enable_background_removal}")
    
    config = DetectionConfig()
    
    # Create pipeline - will use the modified global setting
    pipeline = DataPrepPipeline(
        config=config,
        tenant_id="example_tenant",
        verbose=True
    )
    
    print(f"Pipeline grass mask enabled: {pipeline.enable_grass_mask}")
    print(f"Pipeline steps: {list(pipeline.step_definitions.keys())}")
    
    # Restore original setting
    transform_config.enable_background_removal = original_setting
    print(f"Restored setting: {transform_config.enable_background_removal}")
    print()
    
    return pipeline


def main():
    """Run all examples."""
    print("DataPrepPipeline Grass Mask Configuration Examples")
    print("=" * 55)
    print()
    
    # Example 1: Grass mask enabled
    example_grass_mask_enabled()
    
    # Example 2: Grass mask disabled
    example_grass_mask_disabled()
    
    # Example 3: Default configuration
    example_default_config()
    
    # Example 4: Modify global config
    modify_global_config_example()
    
    print("Summary:")
    print("- Use enable_grass_mask=True to explicitly enable grass mask and background removal")
    print("- Use enable_grass_mask=False to explicitly disable grass mask and background removal")
    print("- Omit enable_grass_mask parameter to use default from transform_config.enable_background_removal")
    print("- When disabled, pipeline skips 'calculate_grass_mask' and 'remove_crop_background' steps")
    print("- When disabled, augmentation works directly with original crops (faster processing)")


if __name__ == "__main__":
    main()
