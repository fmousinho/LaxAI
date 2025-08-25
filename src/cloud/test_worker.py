"""
Test script for Cloud Run Job Worker.

This script can be used to test the worker locally or validate
the deployment in the cloud environment.
"""
import json
import time
import uuid
from typing import Dict, Any

try:
    from google.cloud import pubsub_v1
    from google.cloud import firestore
except ImportError:
    print("Google Cloud libraries not available - install with: pip install google-cloud-pubsub google-cloud-firestore")
    exit(1)


class WorkerTester:
    """Test helper for Cloud Run Job Worker."""
    
    def __init__(self, project_id: str, topic_name: str = "training-jobs"):
        self.project_id = project_id
        self.topic_name = topic_name
        self.publisher = pubsub_v1.PublisherClient()
        self.firestore_client = firestore.Client(project=project_id)
        
    def create_test_job_in_firestore(self, task_id: str) -> Dict[str, Any]:
        """Create a test job directly in Firestore."""
        job_data = {
            "task_id": task_id,
            "status": "queued",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "request_payload": {
                "tenant_id": "test-tenant",
                "training_params": {
                    "epochs": 5,
                    "batch_size": 16,
                    "learning_rate": 0.001
                },
                "model_params": {
                    "architecture": "yolo",
                    "input_size": 640
                },
                "custom_name": f"worker-test-{task_id[:8]}",
                "wandb_tags": ["test", "worker", "cloud-run"]
            },
            "metadata": {
                "source": "test_script",
                "test_mode": True
            }
        }
        
        # Store in Firestore
        doc_ref = self.firestore_client.collection("training_jobs").document(task_id)
        doc_ref.set(job_data)
        
        print(f"✓ Created test job in Firestore: {task_id}")
        return job_data
        
    def publish_test_message(self, task_id: str):
        """Publish a test message to Pub/Sub."""
        topic_path = self.publisher.topic_path(self.project_id, self.topic_name)
        
        message_data = json.dumps({"task_id": task_id}).encode('utf-8')
        
        future = self.publisher.publish(topic_path, message_data)
        message_id = future.result()
        
        print(f"✓ Published message to Pub/Sub: {message_id}")
        return message_id
        
    def monitor_job_progress(self, task_id: str, timeout: int = 300):
        """Monitor job progress in Firestore."""
        print(f"Monitoring job progress for {task_id}...")
        
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            try:
                doc_ref = self.firestore_client.collection("training_jobs").document(task_id)
                doc = doc_ref.get()
                
                if doc.exists:
                    job_data = doc.to_dict()
                    current_status = job_data.get("status")
                    progress = job_data.get("progress", {})
                    
                    if current_status != last_status:
                        print(f"Status: {current_status}")
                        last_status = current_status
                    
                    if progress.get("message"):
                        print(f"Progress: {progress['message']}")
                    
                    if progress.get("progress_percent"):
                        print(f"Progress: {progress['progress_percent']:.1f}%")
                    
                    # Check if job is complete
                    if current_status in ["completed", "failed", "cancelled"]:
                        print(f"✓ Job finished with status: {current_status}")
                        
                        if current_status == "completed":
                            results = progress.get("results", {})
                            print(f"Results: {results}")
                        elif current_status == "failed":
                            error_details = job_data.get("error_details")
                            print(f"Error: {error_details}")
                        
                        return current_status
                        
                else:
                    print(f"✗ Job {task_id} not found in Firestore")
                    return None
                    
            except Exception as e:
                print(f"Error monitoring job: {e}")
                
            time.sleep(5)
        
        print(f"✗ Timeout waiting for job completion ({timeout}s)")
        return "timeout"
        
    def cleanup_test_job(self, task_id: str):
        """Clean up test job from Firestore."""
        try:
            doc_ref = self.firestore_client.collection("training_jobs").document(task_id)
            doc_ref.delete()
            print(f"✓ Cleaned up test job: {task_id}")
        except Exception as e:
            print(f"Error cleaning up job: {e}")


def run_end_to_end_test(project_id: str):
    """Run a complete end-to-end test of the worker system."""
    print("=" * 60)
    print("CLOUD RUN JOB WORKER - END-TO-END TEST")
    print("=" * 60)
    
    tester = WorkerTester(project_id)
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    try:
        # Step 1: Create test job in Firestore
        print("\n1. Creating test job in Firestore...")
        job_data = tester.create_test_job_in_firestore(task_id)
        
        # Step 2: Publish message to trigger worker
        print("\n2. Publishing message to Pub/Sub...")
        message_id = tester.publish_test_message(task_id)
        
        # Step 3: Monitor progress
        print("\n3. Monitoring job progress...")
        final_status = tester.monitor_job_progress(task_id, timeout=600)  # 10 minutes
        
        # Step 4: Report results
        print(f"\n4. Test Results:")
        if final_status == "completed":
            print("✓ END-TO-END TEST PASSED")
        elif final_status == "failed":
            print("✗ END-TO-END TEST FAILED (job failed)")
        elif final_status == "timeout":
            print("✗ END-TO-END TEST FAILED (timeout)")
        else:
            print(f"✗ END-TO-END TEST FAILED (unexpected status: {final_status})")
            
    except Exception as e:
        print(f"✗ END-TO-END TEST FAILED (exception: {e})")
        
    finally:
        # Step 5: Cleanup
        print(f"\n5. Cleaning up...")
        tester.cleanup_test_job(task_id)


def run_load_test(project_id: str, num_jobs: int = 3):
    """Run a load test with multiple concurrent jobs."""
    print("=" * 60)
    print(f"CLOUD RUN JOB WORKER - LOAD TEST ({num_jobs} jobs)")
    print("=" * 60)
    
    tester = WorkerTester(project_id)
    task_ids = []
    
    try:
        # Create and publish multiple jobs
        for i in range(num_jobs):
            task_id = str(uuid.uuid4())
            task_ids.append(task_id)
            
            print(f"\nCreating job {i+1}/{num_jobs}: {task_id}")
            tester.create_test_job_in_firestore(task_id)
            tester.publish_test_message(task_id)
            
            time.sleep(1)  # Slight delay between submissions
        
        # Monitor all jobs
        print(f"\nMonitoring {num_jobs} jobs...")
        completed = 0
        failed = 0
        
        for task_id in task_ids:
            print(f"\nMonitoring job: {task_id}")
            status = tester.monitor_job_progress(task_id, timeout=300)
            
            if status == "completed":
                completed += 1
            elif status == "failed":
                failed += 1
        
        # Results
        print(f"\nLOAD TEST RESULTS:")
        print(f"✓ Completed: {completed}/{num_jobs}")
        print(f"✗ Failed: {failed}/{num_jobs}")
        print(f"? Other: {num_jobs - completed - failed}/{num_jobs}")
        
    finally:
        # Cleanup all jobs
        print(f"\nCleaning up {len(task_ids)} jobs...")
        for task_id in task_ids:
            tester.cleanup_test_job(task_id)


if __name__ == "__main__":
    import sys
    import os
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable not set")
        print("Usage: export GOOGLE_CLOUD_PROJECT=your-project-id")
        exit(1)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "load":
            num_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
            run_load_test(project_id, num_jobs)
        elif sys.argv[1] == "test":
            run_end_to_end_test(project_id)
        else:
            print("Usage: python test_worker.py [test|load] [num_jobs]")
    else:
        run_end_to_end_test(project_id)
    
    print(f"\nTo deploy the worker:")
    print(f"  ./deploy-worker.sh")
    print(f"\nTo test manually:")
    print(f"  gcloud run jobs execute laxai-training-worker --region=us-central1")
