import re
import importlib.util
import os
import pytest


def load_wandb_logger_module():
    path = os.path.join('src', 'train', 'wandb_logger.py')
    spec = importlib.util.spec_from_file_location('wandb_logger_mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sanitize_forbidden_chars_direct():
    mod = load_wandb_logger_module()
    logger = object.__new__(mod.WandbLogger)

    out = logger._sanitize_artifact_name('Dino initial_checkpoint')
    assert out == 'Dino_initial_checkpoint'

    out2 = logger._sanitize_artifact_name('bad/name:with#chars%')
    # Confirm forbidden characters removed
    assert ' ' not in out2 and '/' not in out2 and ':' not in out2 and '#' not in out2 and '%' not in out2
    assert re.match(r'^[A-Za-z0-9._-]+$', out2)


def test_sanitize_leaves_allowed_unchanged():
    mod = load_wandb_logger_module()
    logger = object.__new__(mod.WandbLogger)
    name = 'good-name_1.0'
    assert logger._sanitize_artifact_name(name) == name


def test_get_checkpoint_name_and_sanitize():
    mod = load_wandb_logger_module()
    logger = object.__new__(mod.WandbLogger)

    class DummyRun:
        pass

    dr = DummyRun()
    dr.name = 'Dino initial_checkpoint/with:bad#chars %'
    logger.run = dr
    raw = logger._get_checkpoint_name()
    # raw should contain forbidden characters coming from the run name
    assert any(ch in raw for ch in [' ', '/', ':', '#', '%'])

    safe = logger._sanitize_artifact_name(raw)
    assert re.match(r'^[A-Za-z0-9._-]+$', safe)
