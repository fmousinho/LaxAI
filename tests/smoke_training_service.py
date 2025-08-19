import asyncio
import uuid
import services.training_service as svc


async def main():
    # Monkeypatch the train function to a fast dummy that invokes on_started
    def dummy_train(**kwargs):
        on_started = kwargs.get("on_started")
        run_guid = str(uuid.uuid4())
        if callable(on_started):
            on_started(run_guid)
        # Simulate a short-running run
        return {"run_guid": run_guid, "status": "completed"}

    # Replace the train function used by the service
    svc.train_function = dummy_train

    task_id = await svc.start_job({"tenant_id": "test-tenant", "verbose": False})
    print("scheduled task_id:", task_id)

    # Give the background thread a moment to run the dummy
    await asyncio.sleep(0.2)

    job = svc.get_job(task_id)
    print("job record:", job)

    assert job is not None, "Expected a job record"
    assert job.get("run_guid"), "Expected run_guid to be set on the job"
    # Verify TRAINING_JOBS is keyed by run_guid
    run_guid = job.get("run_guid")
    assert run_guid in svc.TRAINING_JOBS, "Expected TRAINING_JOBS keyed by run_guid"

    print("SMOKE TEST PASS: run_guid captured and TRAINING_JOBS keyed by run_guid")


if __name__ == "__main__":
    asyncio.run(main())
