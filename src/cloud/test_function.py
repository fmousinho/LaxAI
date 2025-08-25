"""
Test script for Cloud Function orchestrator.

This script can be used to test the Cloud Function locally
before deployment to Google Cloud.
"""
import json
import requests
import time
from typing import Dict, Any


class CloudFunctionTester:
    """Test helper for Cloud Function orchestrator."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        
    def submit_training_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a training job to the orchestrator."""
        url = f"{self.base_url}/orchestrate"
        
        response = requests.post(
            url,
            json=job_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"POST {url}")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response.json() if response.status_code < 500 else {"error": response.text}
    
    def get_job_status(self, task_id: str) -> Dict[str, Any]:
        """Get job status from the orchestrator."""
        url = f"{self.base_url}/status"
        
        response = requests.get(
            url,
            params={"task_id": task_id}
        )
        
        print(f"GET {url}?task_id={task_id}")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response.json() if response.status_code < 500 else {"error": response.text}


def create_sample_training_request() -> Dict[str, Any]:
    """Create a sample training request for testing."""
    return {
        "tenant_id": "test-tenant",
        "training_params": {
            "model_type": "detection",
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.001
        },
        "model_params": {
            "architecture": "yolo",
            "input_size": 640
        },
        "custom_name": "test-training-job",
        "wandb_tags": ["test", "cloud-function"]
    }


def run_basic_test():
    """Run basic functionality test."""
    print("=" * 60)
    print("CLOUD FUNCTION ORCHESTRATOR - BASIC TEST")
    print("=" * 60)
    
    tester = CloudFunctionTester()
    
    # Test 1: Submit valid training job
    print("\n1. Testing valid job submission...")
    valid_request = create_sample_training_request()
    
    result = tester.submit_training_job(valid_request)
    
    if "task_id" in result:
        task_id = result["task_id"]
        print(f"✓ Job submitted successfully: {task_id}")
        
        # Test 2: Check job status
        print(f"\n2. Testing job status retrieval...")
        time.sleep(1)  # Give it a moment
        
        status_result = tester.get_job_status(task_id)
        if "status" in status_result:
            print(f"✓ Job status retrieved: {status_result['status']}")
        else:
            print(f"✗ Failed to get job status: {status_result}")
    else:
        print(f"✗ Job submission failed: {result}")
    
    # Test 3: Invalid request (missing required field)
    print(f"\n3. Testing invalid job submission...")
    invalid_request = create_sample_training_request()
    del invalid_request["tenant_id"]  # Remove required field
    
    result = tester.submit_training_job(invalid_request)
    if "error" in result:
        print(f"✓ Invalid request properly rejected: {result['error']}")
    else:
        print(f"✗ Invalid request was incorrectly accepted: {result}")
    
    # Test 4: Non-existent job status
    print(f"\n4. Testing non-existent job status...")
    result = tester.get_job_status("non-existent-job-id")
    if "error" in result:
        print(f"✓ Non-existent job properly handled: {result['error']}")
    else:
        print(f"✗ Non-existent job returned unexpected result: {result}")


def run_load_test(num_jobs: int = 5):
    """Run a simple load test with multiple jobs."""
    print("=" * 60)
    print(f"CLOUD FUNCTION ORCHESTRATOR - LOAD TEST ({num_jobs} jobs)")
    print("=" * 60)
    
    tester = CloudFunctionTester()
    task_ids = []
    
    # Submit multiple jobs
    for i in range(num_jobs):
        print(f"\nSubmitting job {i+1}/{num_jobs}...")
        
        job_data = create_sample_training_request()
        job_data["custom_name"] = f"load-test-job-{i+1}"
        job_data["wandb_tags"] = ["load-test", f"job-{i+1}"]
        
        result = tester.submit_training_job(job_data)
        if "task_id" in result:
            task_ids.append(result["task_id"])
            print(f"✓ Job {i+1} submitted: {result['task_id']}")
        else:
            print(f"✗ Job {i+1} failed: {result}")
    
    # Check status of all jobs
    print(f"\nChecking status of {len(task_ids)} jobs...")
    for i, task_id in enumerate(task_ids):
        result = tester.get_job_status(task_id)
        if "status" in result:
            print(f"✓ Job {i+1} status: {result['status']}")
        else:
            print(f"✗ Job {i+1} status check failed: {result}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        num_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        run_load_test(num_jobs)
    else:
        run_basic_test()
    
    print(f"\nTo run the local Cloud Function server:")
    print(f"  cd src/cloud && python function_orchestrator.py")
    print(f"\nTo run load test:")
    print(f"  python {__file__} load [num_jobs]")
