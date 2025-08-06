#!/usr/bin/env python3
"""
Debug delimiter functionality in list_blobs
"""
import sys
import os
sys.path.insert(0, 'src')

from common.google_storage import get_storage

def debug_delimiter():
    print("Debugging delimiter functionality...")
    
    try:
        # Initialize storage client
        storage_client = get_storage('tenant1')
        
        # Manually test the GCS API directly
        print("\n=== Direct GCS API test ===")
        client = storage_client._client
        bucket = storage_client._bucket
        
        # Ensure authentication
        storage_client._ensure_authenticated()
        
        print(f"Client: {client}")
        print(f"Bucket: {bucket}")
        
        # Test with different prefixes
        test_prefixes = [
            "tenant1/datasets/",
            "tenant1/datasets",
            "datasets/",
            "datasets"
        ]
        
        for prefix in test_prefixes:
            print(f"\n--- Testing prefix: '{prefix}' with delimiter='/' ---")
            try:
                iterator = client.list_blobs(bucket, prefix=prefix, delimiter='/')
                
                # Try to access prefixes directly
                prefixes_list = list(iterator.prefixes)
                print(f"Prefixes count: {len(prefixes_list)}")
                if prefixes_list:
                    print(f"Sample prefixes: {prefixes_list[:3]}")
                else:
                    print("No prefixes found")
                    
                # Also check actual blobs
                iterator2 = client.list_blobs(bucket, prefix=prefix, delimiter='/')
                blobs_list = list(iterator2)
                print(f"Blobs count: {len(blobs_list)}")
                if blobs_list:
                    print(f"Sample blobs: {[b.name for b in blobs_list[:3]]}")
                else:
                    print("No blobs found")
                    
            except Exception as e:
                print(f"Error with prefix '{prefix}': {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_delimiter()
