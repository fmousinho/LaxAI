#!/usr/bin/env python3
"""
Test script to verify grass mask configuration functionality in DataPrepPipeline.
"""

import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.config.all_config import DetectionConfig, transform_config
from src.train.dataprep_pipeline import DataPrepPipeline


def test_grass_mask_enabled():
    """Test pipeline with grass mask enabled."""
    print("Testing DataPrepPipeline with grass mask ENABLED...")
    
    config = DetectionConfig()
    
    try:
        pipeline = DataPrepPipeline(
            config=config, 
            tenant_id="test_tenant",
            enable_grass_mask=True,
            verbose=True
        )
        
        print(f"✓ Pipeline initialized successfully with grass mask enabled: {pipeline.enable_grass_mask}")
        print(f"✓ Background mask detector initialized: {pipeline.background_mask_detector is not None}")
        
        # Check if grass mask steps are included
        step_names = list(pipeline.step_definitions.keys())
        print(f"✓ Pipeline steps: {step_names}")
        
        has_grass_mask = "calculate_grass_mask" in step_names
        has_background_removal = "remove_crop_background" in step_names
        
        print(f"✓ Has grass mask step: {has_grass_mask}")
        print(f"✓ Has background removal step: {has_background_removal}")
        
        if has_grass_mask and has_background_removal:
            print("✓ PASS: Grass mask steps correctly included when enabled\n")
            return True
        else:
            print("✗ FAIL: Grass mask steps missing when enabled\n")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: Exception during pipeline initialization: {e}\n")
        return False


def test_grass_mask_disabled():
    """Test pipeline with grass mask disabled."""
    print("Testing DataPrepPipeline with grass mask DISABLED...")
    
    config = DetectionConfig()
    
    try:
        pipeline = DataPrepPipeline(
            config=config, 
            tenant_id="test_tenant",
            enable_grass_mask=False,
            verbose=True
        )
        
        print(f"✓ Pipeline initialized successfully with grass mask disabled: {pipeline.enable_grass_mask}")
        print(f"✓ Background mask detector not initialized: {pipeline.background_mask_detector is None}")
        
        # Check if grass mask steps are excluded
        step_names = list(pipeline.step_definitions.keys())
        print(f"✓ Pipeline steps: {step_names}")
        
        has_grass_mask = "calculate_grass_mask" in step_names
        has_background_removal = "remove_crop_background" in step_names
        
        print(f"✓ Has grass mask step: {has_grass_mask}")
        print(f"✓ Has background removal step: {has_background_removal}")
        
        if not has_grass_mask and not has_background_removal:
            print("✓ PASS: Grass mask steps correctly excluded when disabled\n")
            return True
        else:
            print("✗ FAIL: Grass mask steps present when disabled\n")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: Exception during pipeline initialization: {e}\n")
        return False


def test_grass_mask_default_config():
    """Test pipeline with default config (should use transform_config.enable_background_removal)."""
    print("Testing DataPrepPipeline with DEFAULT config...")
    
    config = DetectionConfig()
    
    try:
        # Test with default transform_config setting
        original_setting = transform_config.enable_background_removal
        print(f"✓ Current transform_config.enable_background_removal: {original_setting}")
        
        pipeline = DataPrepPipeline(
            config=config, 
            tenant_id="test_tenant",
            verbose=True  # No enable_grass_mask parameter - should use default
        )
        
        print(f"✓ Pipeline initialized with grass mask setting: {pipeline.enable_grass_mask}")
        print(f"✓ Should match transform_config setting: {pipeline.enable_grass_mask == original_setting}")
        
        # Check if grass mask steps match the config
        step_names = list(pipeline.step_definitions.keys())
        has_grass_mask = "calculate_grass_mask" in step_names
        has_background_removal = "remove_crop_background" in step_names
        
        expected_steps = original_setting
        actual_steps = has_grass_mask and has_background_removal
        
        if expected_steps == actual_steps:
            print("✓ PASS: Pipeline steps correctly match default config\n")
            return True
        else:
            print(f"✗ FAIL: Pipeline steps ({actual_steps}) don't match config ({expected_steps})\n")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: Exception during pipeline initialization: {e}\n")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING GRASS MASK CONFIGURATION IN DATAPREP PIPELINE")
    print("=" * 60)
    
    results = []
    
    # Test grass mask enabled
    results.append(test_grass_mask_enabled())
    
    # Test grass mask disabled  
    results.append(test_grass_mask_disabled())
    
    # Test default config
    results.append(test_grass_mask_default_config())
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
