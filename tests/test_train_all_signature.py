import inspect

from src.scripts.train_all import train


def test_train_signature_has_n_datasets_to_use():
    sig = inspect.signature(train)
    assert "n_datasets_to_use" in sig.parameters
