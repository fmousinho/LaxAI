#!/usr/bin/env python3
"""
Quick test to verify the list_blobs fix works properly
"""
import sys
import os
sys.path.insert(0, 'src')

from common.google_storage import get_storage

def test_list_blobs():
    print("Testing list_blobs fix...")
    
    try:
        # Initialize storage client
        storage_client = get_storage('tenant1')
        
        # Test 1: List with delimiter (should return directories as list)
        print("\n=== Test 1: List with delimiter ===")
        datasets = storage_client.list_blobs(prefix="datasets", delimiter='/')
        print(f"Type: {type(datasets)}")
        print(f"Length: {len(datasets)}")
        print(f"First few: {datasets[:3] if len(datasets) > 0 else 'None'}")
        
        # Test 2: List without delimiter (should return files as list)
        print("\n=== Test 2: List without delimiter (limited) ===")
        files = storage_client.list_blobs(prefix="datasets")
        print(f"Type: {type(files)}")
        print(f"Length: {len(files)}")
        print(f"First few: {files[:3] if len(files) > 0 else 'None'}")
        
        print("\n✅ Fix successful! list_blobs now returns lists that can be sliced and have len()")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_list_blobs()
