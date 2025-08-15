"""
Example client for testing the LaxAI Training API.
"""
import asyncio
import json
import httpx
from typing import Dict, Any


class LaxAIClient:
    """
    Simple client for interacting with the LaxAI Training API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def start_training(self, training_request: Dict[str, Any]) -> Dict[str, Any]:
        """Start a training job."""
        response = await self.client.post(
            f"{self.base_url}/api/v1/train",
            json=training_request
        )
        response.raise_for_status()
        return response.json()
    
    async def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """Get training job status."""
        response = await self.client.get(
            f"{self.base_url}/api/v1/train/{task_id}/status"
        )
        response.raise_for_status()
        return response.json()
    
    async def list_training_jobs(self) -> Dict[str, Any]:
        """List all training jobs."""
        response = await self.client.get(
            f"{self.base_url}/api/v1/train/jobs"
        )
        response.raise_for_status()
        return response.json()
    
    async def cancel_training_job(self, task_id: str) -> Dict[str, Any]:
        """Cancel a training job."""
        response = await self.client.delete(
            f"{self.base_url}/api/v1/train/{task_id}"
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()


async def example_usage():
    """
    Example usage of the LaxAI Training API.
    """
    client = LaxAIClient()
    
    try:
        # Example training request
        training_request = {
            "tenant_id": "tenant1",
            "verbose": True,
            "save_intermediate": True,
            "custom_name": "api_test_run",
            "resume_from_checkpoint": True,
            "wandb_tags": ["api-test", "example"],
            "training_kwargs": {
                "num_epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "early_stopping_patience": 5,
                "margin": 0.4
            },
            "model_kwargs": {
                "embedding_dim": 256,
                "dropout_rate": 0.2,
                "use_cbam": True
            }
        }
        
        print("🚀 Starting training job...")
        print(f"Request: {json.dumps(training_request, indent=2)}")
        
        # Start training
        start_response = await client.start_training(training_request)
        print(f"✅ Training started: {json.dumps(start_response, indent=2)}")
        
        task_id = start_response.get("task_id")
        if not task_id:
            print("❌ No task ID returned")
            return
        
        # Check status a few times
        for i in range(3):
            await asyncio.sleep(2)
            print(f"\n📊 Checking status (attempt {i+1})...")
            status_response = await client.get_training_status(task_id)
            print(f"Status: {json.dumps(status_response, indent=2)}")
        
        # List all jobs
        print("\n📋 Listing all training jobs...")
        jobs_response = await client.list_training_jobs()
        print(f"Jobs: {json.dumps(jobs_response, indent=2)}")
        
        # Note: In a real scenario, you might want to cancel long-running jobs
        # print(f"\n🛑 Cancelling job {task_id}...")
        # cancel_response = await client.cancel_training_job(task_id)
        # print(f"Cancel response: {json.dumps(cancel_response, indent=2)}")
        
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    print("🧪 LaxAI Training API Example Client")
    print("=" * 50)
    asyncio.run(example_usage())
