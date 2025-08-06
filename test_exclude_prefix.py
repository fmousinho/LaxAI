#!/usr/bin/env python3
"""
Test script to verify exclude_prefix_in_return functionality
"""
import sys
import os
sys.path.insert(0, 'src')

from common.google_storage import get_storage

def test_exclude_prefix_functionality():
    print("Testing exclude_prefix_in_return functionality...")
    
    try:
        # Initialize storage client
        storage_client = get_storage('tenant1')
        
        print("\n=== Test 1: With delimiter, exclude_prefix_in_return=False (default) ===")
        datasets_normal = storage_client.list_blobs(prefix="datasets", delimiter='/', exclude_prefix_in_return=False)
        print(f"Type: {type(datasets_normal)}")
        print(f"Length: {len(datasets_normal)}")
        print(f"First few: {list(datasets_normal)[:3] if len(datasets_normal) > 0 else 'None'}")
        
        print("\n=== Test 2: With delimiter, exclude_prefix_in_return=True ===")
        datasets_stripped = storage_client.list_blobs(prefix="datasets", delimiter='/', exclude_prefix_in_return=True)
        print(f"Type: {type(datasets_stripped)}")
        print(f"Length: {len(datasets_stripped)}")
        print(f"First few: {list(datasets_stripped)[:3] if len(datasets_stripped) > 0 else 'None'}")
        
        print("\n=== Test 3: Without delimiter, exclude_prefix_in_return=False ===")
        files_normal = storage_client.list_blobs(prefix="datasets", exclude_prefix_in_return=False)
        print(f"Type: {type(files_normal)}")
        print(f"Length: {len(files_normal)}")
        print(f"First few: {list(files_normal)[:3] if len(files_normal) > 0 else 'None'}")
        
        print("\n=== Test 4: Without delimiter, exclude_prefix_in_return=True ===")
        files_stripped = storage_client.list_blobs(prefix="datasets", exclude_prefix_in_return=True)
        print(f"Type: {type(files_stripped)}")
        print(f"Length: {len(files_stripped)}")
        print(f"First few: {list(files_stripped)[:3] if len(files_stripped) > 0 else 'None'}")
        
        # Test specific scenarios to verify the stripping logic
        print("\n=== Test 5: Comparing results ===")
        print("Normal results should include full paths")
        print("Stripped results should have prefixes removed")
        
        if len(datasets_normal) > 0 and len(datasets_stripped) > 0:
            normal_sample = list(datasets_normal)[0]
            stripped_sample = list(datasets_stripped)[0]
            print(f"Normal sample: '{normal_sample}'")
            print(f"Stripped sample: '{stripped_sample}'")
            print(f"Difference: Normal should be longer and contain prefix")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exclude_prefix_functionality()
