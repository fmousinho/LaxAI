#!/usr/bin/env python3
"""
Alternative approach: Extract dataset names from file paths instead of using delimiter
"""
import sys
import os
sys.path.insert(0, 'src')

from common.google_storage import get_storage

def test_alternative_approach():
    print("=== Testing Alternative Approach: Parse Dataset Names from Paths ===")
    
    try:
        storage_client = get_storage('tenant1')
        
        # Ensure authentication first
        if not storage_client._ensure_authenticated():
            print("❌ Failed to authenticate")
            return
            
        print(f"✅ Authenticated successfully")
        
        # Get all files with tenant1/datasets/ prefix (no delimiter)
        iterator = storage_client._bucket.list_blobs(prefix="tenant1/datasets/")
        
        # Extract unique dataset names
        dataset_names = set()
        count = 0
        
        for blob in iterator:
            count += 1
            # Extract dataset name from path like: tenant1/datasets/dataset_0038b37c/train/...
            path_parts = blob.name.split('/')
            if len(path_parts) >= 3 and path_parts[0] == 'tenant1' and path_parts[1] == 'datasets':
                dataset_name = path_parts[2]  # This should be like 'dataset_0038b37c'
                if dataset_name.startswith('dataset_'):
                    dataset_names.add(dataset_name)
            
            # Stop after checking some files to get dataset names
            if count >= 1000:  # Check first 1000 files to find dataset patterns
                break
        
        print(f"Found {len(dataset_names)} unique datasets from {count} files:")
        for i, dataset in enumerate(sorted(dataset_names)[:10]):
            print(f"  {i+1}: {dataset}")
        
        if len(dataset_names) > 10:
            print(f"  ... and {len(dataset_names) - 10} more")
        
        # Now let's implement this approach in our list_blobs method
        print(f"\n=== This is the approach we should use instead of delimiter ===")
        print("✅ Alternative approach: Parse dataset names from file paths")
        print("   This gives us the actual dataset directories we need!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alternative_approach()
