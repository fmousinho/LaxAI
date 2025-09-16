"""
Test script for the Pub/Sub Job Proxy.
"""

import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from main import TrainingJobProxy


def test_message_parsing():
    """Test message parsing logic."""
    proxy = TrainingJobProxy()

    # Test create message
    create_message = {
        "action": "create",
        "tenant_id": "test_tenant",
        "custom_name": "test_run",
        "training_params": {"batch_size": 128},
        "model_params": {"embedding_dim": 256},
        "eval_params": {},
        "resume_from_checkpoint": True
    }

    try:
        parsed = proxy.parse_message(json.dumps(create_message).encode('utf-8'))
        print("✅ Create message parsed successfully")
        assert parsed["action"] == "create"
        assert parsed["tenant_id"] == "test_tenant"
    except Exception as e:
        print(f"❌ Create message parsing failed: {e}")
        return False

    # Test cancel message
    cancel_message = {
        "action": "cancel",
        "job_id": "test-job-123"
    }

    try:
        parsed = proxy.parse_message(json.dumps(cancel_message).encode('utf-8'))
        print("✅ Cancel message parsed successfully")
        assert parsed["action"] == "cancel"
        assert parsed["job_id"] == "test-job-123"
    except Exception as e:
        print(f"❌ Cancel message parsing failed: {e}")
        return False

    # Test invalid messages
    invalid_messages = [
        {"tenant_id": "test"},  # missing action
        {"action": "invalid"},  # invalid action
        {"action": "create"},  # missing tenant_id
        {"action": "cancel"},  # missing job_id
    ]

    for i, msg in enumerate(invalid_messages):
        try:
            proxy.parse_message(json.dumps(msg).encode('utf-8'))
            print(f"❌ Invalid message {i+1} should have failed but didn't")
            return False
        except ValueError:
            print(f"✅ Invalid message {i+1} correctly rejected")
        except Exception as e:
            print(f"❌ Invalid message {i+1} failed with unexpected error: {e}")
            return False

    return True

def test_job_creation_args():
    """Test job creation argument building."""
    proxy = TrainingJobProxy()

    message_data = {
        "tenant_id": "test_tenant",
        "custom_name": "test_run",
        "training_params": {"batch_size": 128, "num_epochs": 10},
        "model_params": {"embedding_dim": 256},
        "eval_params": {"threshold": 0.7},
        "resume_from_checkpoint": True
    }

    job = proxy.create_job_request(message_data)

    # Check that args contain expected values
    args = job.template.template.containers[0].args
    print(f"Generated args: {args}")

    expected_args = [
        "--tenant_id", "test_tenant",
        "--custom_name", "test_run",
        "--resume_from_checkpoint",
        "--batch_size", "128",
        "--num_epochs", "10",
        "--model_embedding_dim", "256",
        "--eval_threshold", "0.7"
    ]

    if args == expected_args:
        print("✅ Job args generated correctly")
        return True
    else:
        print(f"❌ Job args mismatch. Expected: {expected_args}, Got: {args}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Pub/Sub Job Proxy...")

    success = True
    success &= test_message_parsing()
    success &= test_job_creation_args()

    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)