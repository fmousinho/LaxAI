"""
LaxAI Cloud Training Client

A simple Python client for submitting and monitoring GPU training jobs
on Google Cloud using the LaxAI training infrastructure.
"""
import os
import time
import json
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrainingJobRequest:
    """Training job request structure."""
    tenant_id: str
    training_params: Dict[str, Any]
    model_params: Dict[str, Any]
    custom_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "tenant_id": self.tenant_id,
            "training_params": self.training_params,
            "model_params": self.model_params,
            "custom_name": self.custom_name,
            "wandb_tags": self.wandb_tags or []
        }


@dataclass
class JobStatus:
    """Training job status information."""
    task_id: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None


class LaxAITrainingClient:
    """Client for LaxAI cloud training services."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the training client.
        
        Args:
            base_url: Base URL of the Cloud Function orchestrator
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def submit_job(self, job_request: TrainingJobRequest) -> Dict[str, Any]:
        """
        Submit a training job for execution.
        
        Args:
            job_request: Training job configuration
            
        Returns:
            Dictionary with task_id and status
            
        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}/orchestrate"
        
        try:
            response = self.session.post(
                url,
                json=job_request.to_dict(),
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to submit job: {e}")
    
    def get_job_status(self, task_id: str) -> JobStatus:
        """
        Get the status of a training job.
        
        Args:
            task_id: Job identifier
            
        Returns:
            JobStatus object with current job information
            
        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}/status"
        
        try:
            response = self.session.get(
                url,
                params={"task_id": task_id},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            return JobStatus(
                task_id=data["task_id"],
                status=data["status"],
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at"),
                started_at=data.get("started_at"),
                completed_at=data.get("completed_at"),
                progress=data.get("progress"),
                error_details=data.get("error_details")
            )
            
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to get job status: {e}")
    
    def wait_for_completion(
        self, 
        task_id: str, 
        poll_interval: int = 30,
        timeout: int = 7200,
        callback: Optional[callable] = None
    ) -> JobStatus:
        """
        Wait for a job to complete, polling for status updates.
        
        Args:
            task_id: Job identifier
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds
            callback: Optional function called with each status update
            
        Returns:
            Final JobStatus when job completes
            
        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(task_id)
            
            if callback:
                callback(status)
            
            if status.status in ["completed", "failed", "cancelled"]:
                return status
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Job {task_id} did not complete within {timeout} seconds")
    
    def list_jobs(self) -> List[JobStatus]:
        """
        List recent training jobs (if supported by the API).
        
        Note: This would require additional Cloud Function endpoint
        """
        # This would need to be implemented in the Cloud Function
        raise NotImplementedError("Job listing not yet implemented")


def create_sample_job() -> TrainingJobRequest:
    """Create a sample training job for testing."""
    return TrainingJobRequest(
        tenant_id="demo-tenant",
        training_params={
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 0.001,
            "model_type": "detection"
        },
        model_params={
            "architecture": "yolo",
            "input_size": 640,
            "num_classes": 80
        },
        custom_name="demo-training-job",
        wandb_tags=["demo", "client-test", "gpu"]
    )


def main():
    """Command-line interface for the training client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LaxAI Cloud Training Client")
    parser.add_argument("--url", required=True, help="Cloud Function base URL")
    parser.add_argument("--action", choices=["submit", "status", "wait", "demo"], 
                       default="demo", help="Action to perform")
    parser.add_argument("--task-id", help="Task ID for status/wait actions")
    parser.add_argument("--config", help="JSON file with job configuration")
    parser.add_argument("--timeout", type=int, default=7200, 
                       help="Timeout for wait action (seconds)")
    
    args = parser.parse_args()
    
    client = LaxAITrainingClient(args.url)
    
    if args.action == "demo":
        # Submit a demo job
        print("Submitting demo training job...")
        job = create_sample_job()
        
        try:
            result = client.submit_job(job)
            task_id = result["task_id"]
            print(f"✓ Job submitted successfully!")
            print(f"  Task ID: {task_id}")
            print(f"  Status: {result['status']}")
            
            # Monitor progress
            print(f"\nMonitoring job progress...")
            
            def progress_callback(status):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status.status}")
                if status.progress and status.progress.get("message"):
                    print(f"  {status.progress['message']}")
                if status.progress and status.progress.get("progress_percent"):
                    print(f"  Progress: {status.progress['progress_percent']:.1f}%")
            
            final_status = client.wait_for_completion(
                task_id, 
                poll_interval=30,
                timeout=args.timeout,
                callback=progress_callback
            )
            
            print(f"\n✓ Job completed with status: {final_status.status}")
            if final_status.progress and final_status.progress.get("results"):
                print(f"Results: {final_status.progress['results']}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    elif args.action == "submit":
        if not args.config:
            print("Error: --config required for submit action")
            return
        
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        job = TrainingJobRequest(**config)
        result = client.submit_job(job)
        print(f"Job submitted: {result}")
    
    elif args.action == "status":
        if not args.task_id:
            print("Error: --task-id required for status action")
            return
        
        status = client.get_job_status(args.task_id)
        print(f"Status: {status.status}")
        print(f"Progress: {status.progress}")
    
    elif args.action == "wait":
        if not args.task_id:
            print("Error: --task-id required for wait action")
            return
        
        def progress_callback(status):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status.status}")
            if status.progress:
                print(f"  {status.progress}")
        
        final_status = client.wait_for_completion(
            args.task_id,
            timeout=args.timeout,
            callback=progress_callback
        )
        print(f"Final status: {final_status.status}")


if __name__ == "__main__":
    main()
