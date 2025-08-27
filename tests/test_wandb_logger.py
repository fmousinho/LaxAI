import os
import importlib
import pytest

from train.wandb_logger import WandbLogger


def test_sanitize_artifact_name_basic():
    logger = WandbLogger(enabled=False)
    # common unsafe name
    assert logger._sanitize_artifact_name('My Model v1!@#') == 'My_Model_v1'
    # name that becomes empty after sanitization
    assert logger._sanitize_artifact_name('!!!') == 'artifact'
    # preserve allowed characters
    assert logger._sanitize_artifact_name('model.name-01_ok') == 'model.name-01_ok'


def test_get_checkpoint_name_with_run():
    wl = WandbLogger(enabled=False)
    # simulate an initialized run object with a name
    wl.run = type('R', (), {'name': 'test_run'})()
    assert wl.get_checkpoint_name() == 'test_run_checkpoint'


def test_requires_wandb_enabled_raises():
    wl = WandbLogger(enabled=False)
    # init_run is decorated with requires_wandb_enabled -> should raise when disabled
    with pytest.raises(RuntimeError):
        wl.init_run(config={})
