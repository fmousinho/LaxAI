#!/usr/bin/env python3
"""
Test GCS delimiter functionality properly
"""
import sys
import os
sys.path.insert(0, 'src')

from common.google_storage import get_storage

def test_delimiter_properly():
    print("Testing GCS delimiter functionality...")
    
    try:
        storage_client = get_storage('tenant1')
        
        if not storage_client._ensure_authenticated():
            print("❌ Failed to authenticate")
            return
            
        print("✅ Authenticated successfully")
        
        # Test with the exact GCS API pattern
        print("\n=== Testing delimiter functionality ===")
        
        # Get iterator with delimiter
        iterator = storage_client._bucket.list_blobs(prefix="tenant1/datasets/", delimiter="/")
        
        # The key insight: we need to consume the iterator to populate prefixes
        print("Before consuming iterator:")
        print(f"  iterator.prefixes: {list(iterator.prefixes)}")
        
        # Consume some of the iterator 
        blob_count = 0
        for blob in iterator:
            blob_count += 1
            if blob_count >= 10:  # Just consume a few items
                break
        
        print("After consuming some of iterator:")
        print(f"  iterator.prefixes: {list(iterator.prefixes)}")
        print(f"  Blob count processed: {blob_count}")
        
        # Try a fresh iterator and consume all
        print("\n=== Testing with fresh iterator (consume all) ===")
        iterator2 = storage_client._bucket.list_blobs(prefix="tenant1/datasets/", delimiter="/")
        
        # Consume the entire iterator
        all_blobs = list(iterator2)
        print(f"Total blobs: {len(all_blobs)}")
        print(f"Prefixes after full consumption: {list(iterator2.prefixes)}")
        
        # Test different prefix levels
        print("\n=== Testing different prefix levels ===")
        
        # Test tenant1/ level
        iterator3 = storage_client._bucket.list_blobs(prefix="tenant1/", delimiter="/")
        list(iterator3)  # Consume all
        print(f"tenant1/ prefixes: {list(iterator3.prefixes)}")
        
        # Test tenant1/datasets/ level
        iterator4 = storage_client._bucket.list_blobs(prefix="tenant1/datasets/", delimiter="/")
        list(iterator4)  # Consume all
        print(f"tenant1/datasets/ prefixes: {list(iterator4.prefixes)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_delimiter_properly()
