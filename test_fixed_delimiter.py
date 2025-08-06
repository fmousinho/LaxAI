#!/usr/bin/env python3
"""
Test the fixed list_blobs delimiter functionality
"""
import sys
import os
sys.path.insert(0, 'src')

from common.google_storage import get_storage

def test_fixed_list_blobs():
    print("Testing FIXED list_blobs delimiter functionality...")
    
    try:
        storage_client = get_storage('tenant1')
        
        # Test 1: List with delimiter (should return directories as set)
        print("\n=== Test 1: List with delimiter (FIXED) ===")
        datasets = storage_client.list_blobs(prefix="datasets", delimiter='/')
        print(f"Type: {type(datasets)}")
        print(f"Length: {len(datasets)}")
        print(f"Sample items: {list(datasets)[:3] if len(datasets) > 0 else 'None'}")
        
        # Test 2: Test exclude_prefix_in_return with delimiter (FIXED)
        print("\n=== Test 2: exclude_prefix_in_return=True with delimiter (FIXED) ===")
        datasets_clean = storage_client.list_blobs(prefix="datasets", delimiter='/', exclude_prefix_in_return=True)
        print(f"Type: {type(datasets_clean)}")
        print(f"Length: {len(datasets_clean)}")
        print(f"Sample items: {list(datasets_clean)[:3] if len(datasets_clean) > 0 else 'None'}")
        
        if len(datasets_clean) > 0:
            print("✅ DELIMITER FUNCTIONALITY NOW WORKING!")
        else:
            print("❌ Still not working")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_list_blobs()
