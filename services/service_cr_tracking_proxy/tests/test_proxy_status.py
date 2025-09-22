import pytest
import os
from unittest.mock import MagicMock, patch
from services.service_cr_tracking_proxy.main import TrackingJobProxy

class DummyDoc:
    def __init__(self, exists, data=None):
        self.exists = exists
        self._data = data or {}
    def to_dict(self):
        return self._data
    def get(self):
        return self
    def update(self, d):
        self._data.update(d)
    def set(self, d):
        self._data = d

class DummyCollection:
    def __init__(self):
        self.docs = {}
    def document(self, task_id):
        if task_id not in self.docs:
            self.docs[task_id] = DummyDoc(False)
        return self.docs[task_id]

@pytest.fixture
def proxy():
    # Ensure GOOGLE_CLOUD_PROJECT is set for the test
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "dummy-project")
    with patch("services.service_cr_tracking_proxy.main.JobsClient", MagicMock()), \
         patch("services.service_cr_tracking_proxy.main.ExecutionsClient", MagicMock()), \
         patch("services.service_cr_tracking_proxy.main.firestore.Client", MagicMock()):
        proxy = TrackingJobProxy()
    # Patch Firestore collection with a MagicMock to match CollectionReference type
    proxy._runs_collection = MagicMock(spec=["document"])
    proxy._runs_collection.document.side_effect = DummyCollection().document
    # Patch run_client and operations_client
    proxy.run_client = MagicMock()
    proxy.run_client.run_job.return_value = MagicMock(
        operation=MagicMock(name='operation', name_='op123'),
        metadata=MagicMock()
    )
    proxy.run_client.run_job.return_value.metadata.name = 'exec123'
    proxy.executions_client = MagicMock()
    return proxy

def test_start_tracking_job_persists_mapping(proxy):
    payload = {"task_id": "abc123", "foo": "bar", "tenant_id": "dummy-tenant"}
    proxy.start_tracking_job(payload)
    doc = proxy._runs_collection.document("abc123")
    assert doc._data["task_id"] == "abc123"
    assert doc._data["execution_name"] == "exec123"
    assert doc._data["status"] == "not_started"
    assert "created_at" in doc._data
    assert "updated_at" in doc._data

def test_cancel_tracking_job_updates_status(proxy):
    # Pre-populate mapping
    doc = proxy._runs_collection.document("abc123")
    doc.exists = True
    doc._data = {"execution_name": "exec123", "status": "running"}
    proxy.cancel_tracking_job("abc123")
    assert doc._data["status"] == "cancelled"
    assert "updated_at" in doc._data

def test_get_tracking_job_status(proxy):
    doc = proxy._runs_collection.document("abc123")
    doc.exists = True
    doc._data = {"task_id": "abc123", "status": "running"}
    status = proxy.get_tracking_job_status("abc123")
    assert status["task_id"] == "abc123"
    assert status["status"] == "running"
    # Not found
    status2 = proxy.get_tracking_job_status("notfound")
    assert status2["status"] == "not_found"
    assert "error" in status2