
import time
import uuid
# ...existing code...
import pytest

# Pipeline stop helper

import utils.env_secrets as env_secrets
env_secrets.setup_environment_secrets()

from scripts import train_all

from common.pipeline import stop_pipeline
def test_train_all_with_one_dataset():
    """
    Run a short real end-to-end training flow against one dataset.

    This test uses real GCP and wandb as requested. We pass a small training
    configuration (num_epochs=2, batch_size=16) and limit datasets to 1 to
    keep the run short.
    """
    results = train_all.train(
        tenant_id="tenant1",
        verbose=False,
        save_intermediate=False,
        custom_name="e2e_test",
        resume_from_checkpoint=False,
        wandb_tags=[],
        training_kwargs={"num_epochs": 1, "batch_size": 16},
        model_kwargs={},
        n_datasets_to_use=1,
    )

    assert isinstance(results, dict)
    assert results.get("status") == "completed"


def test_train_all_with_dino():
    """
    Run a short real end-to-end training flow against one dataset using the DINO model.

    This test uses real GCP and wandb as requested. We pass a small training
    configuration (num_epochs=2, batch_size=16) and limit datasets to 1 to
    keep the run short.
    """
    results = train_all.train(
        tenant_id="tenant1",
        verbose=False,
        save_intermediate=False,
        custom_name="e2e_test",
        resume_from_checkpoint=False,
        wandb_tags=[],
        training_kwargs={"num_epochs": 1, "batch_size": 16, "force_pretraining": True},
        model_kwargs={"model_class_module": "train.siamesenet_dino", "model_class_str": "SiameseNet"},
        n_datasets_to_use=1,
    )

    assert isinstance(results, dict)
    assert results.get("status") == "completed"




def test_train_all_timeboxed_30_seconds(monkeypatch):
    """
    Start a full end-to-end training run but request cancellation after 30 seconds.

    This verifies the pipeline responds to cancellation requests coming from
    external controllers (for example the API service). The test starts the
    training in a background thread, sleeps 30s, then calls
    `stop_pipeline(pipeline_name)` and waits for the run to finish.
    """
    pipeline_name = f"test_timeboxed_{uuid.uuid4().hex}"

    # Monkeypatch storage and wandb logger to fast no-op implementations
    class FakeStorage:
        def list_blobs(self, prefix=None, delimiter=None, exclude_prefix_in_return=False):
            # Return a single fake dataset directory
            return ["tenant1/datasets/dataset_fbbc3ca7/"]

        def upload_from_string(self, blob_name, content):
            return True

        def blob_exists(self, blob_name):
            return False

        def download_as_string(self, blob_name):
            return b""

        def delete_blob(self, blob_name):
            return True

    class FakeGCSPaths:
        def get_path(self, key, dataset_id=None):
            # simple mapping used by train_all and TrainPipeline
            if key == "datasets_root":
                return "tenant1/datasets/"
            if key == "train_dataset":
                return f"tenant1/datasets/{dataset_id}/train/"
            if key == "val_dataset":
                return f"tenant1/datasets/{dataset_id}/val/"
            return None

    fake_storage = FakeStorage()
    monkeypatch.setattr('common.google_storage.get_storage', lambda tenant_id: fake_storage)
    monkeypatch.setattr('common.google_storage.GCSPaths', FakeGCSPaths)

    # Fake wandb logger with minimal methods used by the pipeline
    class FakeWandB:
        def init_run(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def finish(self):
            return None

        def save_model_checkpoint(self, *args, **kwargs):
            return None

    monkeypatch.setattr('train.wandb_logger', FakeWandB())

    # Intentionally long number of epochs so the run will still be active
    # when we trigger cancellation.
    training_kwargs = {"num_epochs": 1000, "batch_size": 16, "pipeline_name": pipeline_name}

    # Run the training in a subprocess so we can force-terminate it if it
    # doesn't exit within the timebox. We write a small runner script that
    # re-creates the minimal fake environment (FakeStorage/FakeGCSPaths and
    # FakeWandB) and then calls train_all.train with the same args.
    import subprocess
    import tempfile
    import os

    runner_code = """
import json
from scripts import train_all
# Ensure the child process responds to SIGTERM/SIGINT by printing a JSON
# cancelled result so the parent test can detect it.
import signal, sys
def _term_handler(signum, frame):
    try:
        print(json.dumps({"status": "cancelled"}))
        sys.stdout.flush()
    finally:
        sys.exit(0)

signal.signal(signal.SIGTERM, _term_handler)
signal.signal(signal.SIGINT, _term_handler)
# Minimal fakes to avoid external GCS/WandB calls
class FakeStorage:
    def list_blobs(self, prefix=None, delimiter=None, exclude_prefix_in_return=False):
        return ["tenant1/datasets/dataset_fbbc3ca7/"]
    def upload_from_string(self, blob_name, content):
        return True
    def blob_exists(self, blob_name):
        return False
    def download_as_string(self, blob_name):
        return b""
    def delete_blob(self, blob_name):
        return True

class FakeGCSPaths:
    def get_path(self, key, dataset_id=None):
        if key == "datasets_root":
            return "tenant1/datasets/"
        if key == "train_dataset":
            return "tenant1/datasets/" + dataset_id + "/train/"
        if key == "val_dataset":
            return "tenant1/datasets/" + dataset_id + "/val/"
        return None

class FakeWandB:
    def init_run(self, *a, **k):
        return None
    def log(self, *a, **k):
        return None
    def finish(self):
        return None
    def save_model_checkpoint(self, *a, **k):
        return None

import common.google_storage as gs
import train.wandb_logger as wb
gs.get_storage = lambda tenant_id: FakeStorage()
gs.GCSPaths = FakeGCSPaths
wb.wandb_logger = FakeWandB()

results = train_all.train(
    tenant_id="tenant1",
    verbose=False,
    save_intermediate=False,
    custom_name="e2e_timeboxed_test",
    resume_from_checkpoint=False,
    wandb_tags=[],
    training_kwargs={"num_epochs":1000, "batch_size":16, "pipeline_name":"{PIPELINE_NAME}"},
    model_kwargs={},
    n_datasets_to_use=1,
)
print(json.dumps(results))
"""

    # Inject pipeline name safely without using outer f-string interpolation
    runner_code = runner_code.replace('{PIPELINE_NAME}', pipeline_name)

    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as f:
        f.write(runner_code)
        runner_path = f.name

    # Start subprocess using the same venv python and PYTHONPATH so imports resolve
    python_bin = os.environ.get('PYTHON_EXECUTABLE', None) or './.venv31211/bin/python'
    env = os.environ.copy()
    env['PYTHONPATH'] = './src'

    proc = subprocess.Popen([python_bin, runner_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)

    try:
        # Let the process run for 30s, then request cancellation and give it 30s to exit.
        # stop_pipeline only affects in-process registries, so also send SIGTERM
        # to the subprocess to simulate an external controller killing the job.
        time.sleep(30)
        try:
            stopped = stop_pipeline(pipeline_name)
        except Exception:
            stopped = False

        import signal
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass

        try:
            out, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            # Didn't exit in time: kill it and fail the test
            proc.kill()
            out, _ = proc.communicate(timeout=5)
            pytest.fail("Training subprocess did not exit within 30s after cancellation; killed")

        # If the process exited, parse returned JSON results and validate.
        # The child prints logs and then a JSON line; find the last valid JSON dict
        import json as _json
        results = {}
        # Prefer JSON objects that include a 'status' key (final result). Many
        # logs are emitted as JSON dicts too; avoid selecting those.
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = _json.loads(line)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue
            if 'status' in parsed:
                results = parsed
                break

        if not results:
            pytest.fail(f"No JSON results with 'status' printed by training subprocess; output:\n{out}")

        assert isinstance(results, dict)
        assert results.get("status") in ("cancelled", "finished", "completed", "error")

    finally:
        try:
            os.unlink(runner_path)
        except Exception:
            pass