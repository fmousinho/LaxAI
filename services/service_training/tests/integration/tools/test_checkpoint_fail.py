#!/usr/bin/env python3
import gc
import os
import sys
import time

# Ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import psutil
import torch
from src.wandb_logger import WandbLogger


class DummyRun:
    def __init__(self, name='test_run'):
        self.name = name
    def log_artifact(self, artifact, type=None, tags=None):
        # simulate an upload failure
        raise RuntimeError('simulated upload failure')
    def finish(self):
        pass

# Create and configure wandb_logger instance for test
wandb_logger = WandbLogger()
wandb_logger.enabled = True
wandb_logger.initialized = True
wandb_logger.run = DummyRun()

proc = psutil.Process()
print('PID:', proc.pid)

mems = []
print('Starting repeated failing checkpoint saves (5 iterations)')
for epoch in range(1, 6):
    gc.collect()
    mem_before = proc.memory_info().rss / 1024 / 1024
    print(f'[{epoch}] mem_before: {mem_before:.1f} MB')
    try:
        wandb_logger.save_checkpoint(epoch=epoch, model_state_dict={'w': torch.tensor([1.0])}, optimizer_state_dict={}, loss=0.1, model_name='test')
        print(f'[{epoch}] save_checkpoint returned normally')
    except Exception as e:
        print(f'[{epoch}] save_checkpoint raised: {repr(e)}')
    # force gc and wait a bit
    gc.collect()
    time.sleep(0.2)
    mem_after = proc.memory_info().rss / 1024 / 1024
    print(f'[{epoch}] mem_after:  {mem_after:.1f} MB (delta {mem_after-mem_before:+.1f} MB)')
    mems.append((mem_before, mem_after))

print('\nSummary:')
for i, (b, a) in enumerate(mems, 1):
    print(f'iter {i}: before={b:.1f}MB after={a:.1f}MB delta={a-b:+.1f}MB')
