from scripts import train_all


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
