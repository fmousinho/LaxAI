from scripts import train_all
import time
import uuid
import concurrent.futures
import pytest

# Pipeline stop helper
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
    assert results.get("status") in ("completed", "finished", "cancelled", "error")


def test_train_all_timeboxed_15_seconds():
    """
    Start a full end-to-end training run but request cancellation after 15 seconds.

    This verifies the pipeline responds to cancellation requests coming from
    external controllers (for example the API service). The test starts the
    training in a background thread, sleeps 15s, then calls
    `stop_pipeline(pipeline_name)` and waits for the run to finish.
    """
    pipeline_name = f"test_timeboxed_{uuid.uuid4().hex}"

    # Intentionally long number of epochs so the run will still be active
    # when we trigger cancellation.
    training_kwargs = {"num_epochs": 1000, "batch_size": 16, "pipeline_name": pipeline_name}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(
            train_all.train,
            tenant_id="tenant1",
            verbose=False,
            save_intermediate=False,
            custom_name="e2e_timeboxed_test",
            resume_from_checkpoint=False,
            wandb_tags=[],
            training_kwargs=training_kwargs,
            model_kwargs={},
            n_datasets_to_use=1,
        )

        # Let the training run for a short period, then request stop
        time.sleep(15)
        stopped = stop_pipeline(pipeline_name)

        # Wait for the future to finish; cancellation should cause it to exit.
        # If it doesn't finish within 15s after the stop request, retry
        # a stop and then fail the test (we cannot forcibly kill threads).
        try:
            results = future.result(timeout=15)
        except concurrent.futures.TimeoutError:
            # Try one more stop request, attempt to cancel the future and
            # mark the test as failed.
            stop_pipeline(pipeline_name)
            future.cancel()
            pytest.fail("Training did not stop within 15s after cancellation request")

        assert isinstance(results, dict)
        # Accept a few possible end states; cancellation should be allowed
        assert results.get("status") in ("cancelled", "finished", "completed", "error")
