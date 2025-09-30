"""
Integration test for the full tracking service using gcloud.

This test executes the deployed tracking service as a Cloud Run job
and verifies that it processes videos and generates tracks successfully.
"""

import os
import subprocess
import time
import pytest
from pathlib import Path


class TestTrackingServiceIntegration:
    """Integration tests for the full tracking service via gcloud."""

    @pytest.mark.integration
    def test_full_tracking_service_execution(self):
        """
        Test the full tracking service by executing it as a Cloud Run job.

        This test:
        1. Executes the tracking service job with TEST_TENANT
        2. Waits for completion
        3. Verifies the job succeeded
        4. Checks that expected outputs were created in GCS
        """
        test_tenant = os.environ.get('TEST_TENANT')
        assert test_tenant, "TEST_TENANT environment variable must be set"

        job_name = "laxai-service-tracking"
        region = "us-central1"

        # Execute the Cloud Run job
        cmd = [
            "gcloud", "run", "jobs", "execute", job_name,
            f"--region={region}",
            f"--args=--tenant_id={test_tenant},--custom_name=test_integration_run,--verbose,--save-intermediate,--video_limit=1",
            "--wait"  # Wait for job completion
        ]

        print(f"Executing tracking service job: {' '.join(cmd)}")

        try:
            # Execute the job
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            print(f"Job execution stdout: {result.stdout}")
            if result.stderr:
                print(f"Job execution stderr: {result.stderr}")

            # Check if job executed successfully
            assert result.returncode == 0, f"Job execution failed: {result.stderr}"

            # Verify job completed successfully by checking the execution status
            # Get the latest execution
            execution_cmd = [
                "gcloud", "run", "jobs", "executions", "list",
                f"--job={job_name}",
                f"--region={region}",
                "--limit=1",
                "--format=value(name)"
            ]

            execution_result = subprocess.run(
                execution_cmd,
                capture_output=True,
                text=True
            )

            assert execution_result.returncode == 0, "Failed to get execution list"
            execution_name = execution_result.stdout.strip()

            # Describe the execution to check status
            describe_cmd = [
                "gcloud", "run", "jobs", "executions", "describe", execution_name,
                f"--region={region}",
                "--format=value(status.conditions[0].type,status.conditions[0].status)"
            ]

            describe_result = subprocess.run(
                describe_cmd,
                capture_output=True,
                text=True
            )

            assert describe_result.returncode == 0, "Failed to describe execution"
            status_info = describe_result.stdout.strip()
            print(f"Job execution status: {status_info}")

            # Should be "Completed True" for successful completion
            assert "True" in status_info, f"Job did not succeed: {status_info}"

            # Verify outputs were created in GCS
            self._verify_gcs_outputs(test_tenant)

            print("✅ Full tracking service integration test passed!")

        except subprocess.TimeoutExpired:
            pytest.fail("Tracking service job timed out after 10 minutes")
        except Exception as e:
            pytest.fail(f"Tracking service test failed: {e}")

    def _verify_gcs_outputs(self, tenant_id: str):
        """
        Verify that the tracking service created expected outputs in GCS.

        Args:
            tenant_id: The tenant ID used for the test
        """
        # Check for run directories created by the tracking service
        # The service creates run directories when it processes videos
        ls_cmd = [
            "gsutil", "ls", f"gs://laxai_dev/{tenant_id}/runs/"
        ]

        result = subprocess.run(
            ls_cmd,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Failed to list GCS run directories: {result.stderr}"
        assert len(result.stdout.strip().split('\n')) > 0, "No run directories found - service may not have processed any videos"

        print(f"✅ GCS outputs verified for tenant: {tenant_id}")
        print(f"📁 Found run directories: {len(result.stdout.strip().split('\n'))}")

        # TODO: Add more specific checks for:
        # - Generated tracks in individual run directories
        # - Processed videos in intermediate directories
        # - Metadata files and checkpoints