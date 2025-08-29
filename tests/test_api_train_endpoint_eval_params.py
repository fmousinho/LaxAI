import sys
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_train_endpoint_forwards_eval_params(monkeypatch):
    """Ensure POST /api/v1/train forwards eval_params and does not start heavy work when patched."""
    sys.path.append('src')

    # Import and patch BEFORE creating the app
    from api.v1.endpoints import train as train_endpoint
    from services import training_service

    captured = {}

    def fake_create_job(request):
        # capture the incoming request object's eval_params (Pydantic model or dict-like)
        eval_params = getattr(request, 'eval_params', None)
        # Convert Pydantic model to dict for comparison
        if eval_params and hasattr(eval_params, 'model_dump'):
            eval_params = eval_params.model_dump(exclude_none=True)
        captured['eval_params'] = eval_params
        # return a fake task id and kwargs that would normally be returned
        return 'fake-task-id', { 'eval_kwargs': eval_params }

    def fake_start_job(task_id, kwargs, background_tasks):
        # don't schedule any background work in the test
        captured['started'] = True

    # Patch the functions on the module used by the router
    monkeypatch.setattr(training_service, 'create_job', fake_create_job)
    monkeypatch.setattr(training_service, 'start_job', fake_start_job)

    app = FastAPI()
    app.include_router(train_endpoint.router, prefix='/api/v1')
    client = TestClient(app)

    payload = {
        "tenant_id": "tenant1",
        "verbose": True,
        "custom_name": "test_run",
        "resume_from_checkpoint": True,
        "wandb_tags": ["t1"],
        "n_datasets_to_use": 0,
        "training_params": {},
        "model_params": {},
        "eval_params": {"emb_batch_size": 4, "number_of_workers": 0}
    }

    resp = client.post('/api/v1/train', json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get('status') == 'accepted'

    # Ensure our fake create_job was called and captured the eval_params
    assert 'eval_params' in captured
    expected = {'number_of_workers': 0, 'emb_batch_size': 4, 'prefetch_factor': 2}
    assert captured['eval_params'] == expected
    assert captured.get('started', False) is True
    assert captured.get('started', False) is True
