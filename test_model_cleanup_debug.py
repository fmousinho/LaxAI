#!/usr/bin/env python3
"""
Debug script to test model artifact cleanup specifically.
"""

import os
import sys
import gc
import tempfile
import uuid
import importlib.util
import torch

# Add src to Python path
sys.path.insert(0, 'src')

def load_wandb_logger_module():
    path = os.path.join('src', 'train', 'wandb_logger.py')
    spec = importlib.util.spec_from_file_location('wandb_logger_mod', path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    print("Testing WandB model artifact cleanup...")
    
    # Load the module
    mod = load_wandb_logger_module()
    
    # Create logger instance
    logger = mod.WandbLogger(enabled=True)
    
    # Initialize test run
    run_name = f"test_model_cleanup_{uuid.uuid4().hex[:8]}"
    logger.init_run({'test': True}, run_name=run_name)
    
    print(f"✅ Initialized test run: {run_name}")
    print(f"✅ Test run detected: {logger._is_test_run()}")
    
    # Create a simple model
    model = torch.nn.Linear(10, 2)
    
    # Test 1: Save a model to registry
    collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
    model_name = logger._get_artifact_name(collection_name)
    print(f"✅ Model will be saved as: {model_name}")
    
    logger.save_model_to_registry(
        model, 
        collection_name=collection_name,
        alias='test',
        metadata={'test': True}
    )
    print(f"✅ Model saved to registry: {collection_name}")
    
    # Test 2: Save another model with different name  
    collection_name2 = f"train.siamesenet_dino_{uuid.uuid4().hex[:6]}"
    model_name2 = logger._get_artifact_name(collection_name2)
    print(f"✅ Second model will be saved as: {model_name2}")
    
    logger.save_model_to_registry(
        model, 
        collection_name=collection_name2,
        alias='test',
        metadata={'test': True}
    )
    print(f"✅ Second model saved: {collection_name2}")
    
    # Wait for any pending async operations
    logger._wait_for_pending_operations()
    print("✅ Async operations completed")
    
    # Test 3: Check artifacts before cleanup
    try:
        artifacts = list(logger.run.logged_artifacts())
        print(f"✅ Found {len(artifacts)} artifacts before cleanup:")
        for artifact in artifacts:
            print(f"   - {artifact.name} (type: {artifact.type})")
    except Exception as e:
        print(f"❌ Could not list artifacts: {e}")
    
    # Test 4: Manual cleanup test
    print("\n🧹 Testing cleanup methods...")
    
    # Test lightweight deletion
    try:
        logger._lightweight_artifact_delete(model_name, "model")
        print(f"✅ Lightweight delete successful for: {model_name}")
    except Exception as e:
        print(f"❌ Lightweight delete failed for {model_name}: {e}")
    
    try:
        logger._lightweight_artifact_delete(model_name2, "model")
        print(f"✅ Lightweight delete successful for: {model_name2}")
    except Exception as e:
        print(f"❌ Lightweight delete failed for {model_name2}: {e}")
    
    # Test 5: Full test cleanup
    print("\n🧹 Testing _cleanup_test_artifacts...")
    try:
        logger._cleanup_test_artifacts()
        print("✅ Test artifacts cleanup completed")
    except Exception as e:
        print(f"❌ Test cleanup failed: {e}")
    
    # Test 6: Check artifacts after cleanup
    try:
        artifacts_after = list(logger.run.logged_artifacts())
        print(f"✅ Found {len(artifacts_after)} artifacts after cleanup:")
        for artifact in artifacts_after:
            print(f"   - {artifact.name} (type: {artifact.type})")
    except Exception as e:
        print(f"❌ Could not list artifacts after cleanup: {e}")
    
    # Test 7: Finish run (should auto-cleanup)
    print("\n🏁 Finishing run with auto-cleanup...")
    logger.finish()
    print("✅ Run finished")
    
    print("\n🎉 Model cleanup debug test completed!")

if __name__ == "__main__":
    main()
