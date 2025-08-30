#!/usr/bin/env python3
"""
Diagnostic script to analyze WandB process spawning during training.
This will help identify why we're running out of CPU memory.
"""

import os
import sys
import time
import psutil
import subprocess
from typing import Dict, List, Set
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.env_secrets import setup_environment_secrets
from train.wandb_logger import WandbLogger
from config.all_config import wandb_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_process_tree_snapshot() -> Dict:
    """Get a snapshot of all processes and their children."""
    current_process = psutil.Process()
    main_pid = current_process.pid

    # Get all child processes recursively
    all_children = []
    try:
        all_children = current_process.children(recursive=True)
    except psutil.NoSuchProcess:
        pass

    # Get all processes that might be WandB related
    wandb_related = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['name'] and 'wandb' in proc.info['name'].lower():
                wandb_related.append(proc)
            elif proc.info['cmdline'] and any('wandb' in str(cmd).lower() for cmd in proc.info['cmdline']):
                wandb_related.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        'main_pid': main_pid,
        'child_processes': len(all_children),
        'wandb_processes': len(wandb_related),
        'total_processes': len(all_children) + len(wandb_related),
        'wandb_process_details': [
            {
                'pid': p.pid,
                'name': p.info['name'],
                'cmdline': p.info['cmdline'][:3] if p.info['cmdline'] else [],  # First 3 args only
                'cpu_percent': p.info['cpu_percent'],
                'memory_mb': p.info['memory_info'].rss / 1024 / 1024 if p.info['memory_info'] else 0,
                'create_time': p.info['create_time']
            } for p in wandb_related
        ]
    }

def analyze_wandb_operations():
    """Analyze WandB operations that spawn processes."""
    logger.info("Setting up environment...")
    setup_environment_secrets()

    logger.info("Taking initial process snapshot...")
    initial_snapshot = get_process_tree_snapshot()
    logger.info(f"Initial state: {initial_snapshot['total_processes']} total processes, {initial_snapshot['wandb_processes']} WandB processes")

    # Initialize WandB
    logger.info("Initializing WandB...")
    wandb_logger = WandbLogger()
    config = {"test": True, "learning_rate": 0.001}
    wandb_logger.init_run(config, run_name="process_diagnostic")

    after_init_snapshot = get_process_tree_snapshot()
    new_processes_init = after_init_snapshot['total_processes'] - initial_snapshot['total_processes']
    logger.info(f"After WandB init: {after_init_snapshot['total_processes']} total processes (+{new_processes_init})")

    # Log metrics multiple times
    logger.info("Logging metrics (simulating epoch logging)...")
    for i in range(5):
        wandb_logger.log_metrics({
            "epoch": i + 1,
            "loss": 1.0 - i * 0.1,
            "accuracy": 0.5 + i * 0.1
        })
        time.sleep(0.1)  # Small delay to let processes spawn

    after_metrics_snapshot = get_process_tree_snapshot()
    new_processes_metrics = after_metrics_snapshot['total_processes'] - after_init_snapshot['total_processes']
    logger.info(f"After metrics logging: {after_metrics_snapshot['total_processes']} total processes (+{new_processes_metrics})")

    # Save checkpoint
    logger.info("Saving checkpoint (simulating epoch checkpoint)...")
    import torch
    mock_model_state = {"layer.weight": torch.randn(10, 5)}
    mock_optimizer_state = {"state": {}, "param_groups": [{"lr": 0.001}]}
    wandb_logger.save_checkpoint(
        1,
        mock_model_state,
        mock_optimizer_state,
        0.5,
        "diagnostic_model"
    )

    after_checkpoint_snapshot = get_process_tree_snapshot()
    new_processes_checkpoint = after_checkpoint_snapshot['total_processes'] - after_metrics_snapshot['total_processes']
    logger.info(f"After checkpoint save: {after_checkpoint_snapshot['total_processes']} total processes (+{new_processes_checkpoint})")

    # Analyze WandB process details
    logger.info("Analyzing WandB process details...")
    for proc_info in after_checkpoint_snapshot['wandb_process_details']:
        age_seconds = time.time() - proc_info['create_time']
        logger.info(f"WandB Process {proc_info['pid']} ({proc_info['name']}): "
                   f"{proc_info['memory_mb']:.1f}MB, {proc_info['cpu_percent']:.1f}% CPU, "
                   f"age: {age_seconds:.1f}s")
        if proc_info['cmdline']:
            logger.info(f"  Command: {' '.join(proc_info['cmdline'])}")

    # Finish WandB
    logger.info("Finishing WandB...")
    wandb_logger.finish()

    final_snapshot = get_process_tree_snapshot()
    logger.info(f"Final state: {final_snapshot['total_processes']} total processes")

    # Summary
    logger.info("=== PROCESS SPAWNING SUMMARY ===")
    logger.info(f"Initial processes: {initial_snapshot['total_processes']}")
    logger.info(f"After WandB init: +{new_processes_init}")
    logger.info(f"After metrics logging: +{new_processes_metrics}")
    logger.info(f"After checkpoint save: +{new_processes_checkpoint}")
    logger.info(f"Total new processes: {final_snapshot['total_processes'] - initial_snapshot['total_processes']}")

    return {
        'initial': initial_snapshot,
        'after_init': after_init_snapshot,
        'after_metrics': after_metrics_snapshot,
        'after_checkpoint': after_checkpoint_snapshot,
        'final': final_snapshot
    }

def check_wandb_settings():
    """Check current WandB settings that might affect process spawning."""
    logger.info("=== WandB CONFIGURATION ANALYSIS ===")

    # Check environment variables
    wandb_env_vars = {k: v for k, v in os.environ.items() if k.startswith('WANDB_')}
    for key, value in wandb_env_vars.items():
        logger.info(f"{key}: {value}")

    # Check if offline mode is enabled
    if 'WANDB_MODE' in os.environ and os.environ['WANDB_MODE'] == 'offline':
        logger.info("✅ WandB is in offline mode - this should reduce process spawning")
    else:
        logger.info("⚠️  WandB is in online mode - this can spawn background sync processes")

    # Check sync settings
    if 'WANDB_DISABLE_SERVICE' in os.environ:
        logger.info("✅ WandB service is disabled")
    else:
        logger.info("⚠️  WandB service is enabled - this spawns background processes")

def main():
    """Main diagnostic function."""
    logger.info("Starting WandB process spawning diagnostic...")

    # Check WandB settings
    check_wandb_settings()

    # Analyze process spawning
    results = analyze_wandb_operations()

    # Recommendations
    logger.info("=== RECOMMENDATIONS ===")

    total_new_processes = results['final']['total_processes'] - results['initial']['total_processes']
    if total_new_processes > 5:
        logger.warning(f"⚠️  HIGH PROCESS COUNT: {total_new_processes} new processes spawned")
        logger.info("Consider enabling WandB offline mode: export WANDB_MODE=offline")
        logger.info("Consider disabling WandB service: export WANDB_DISABLE_SERVICE=true")
        logger.info("Consider reducing checkpoint frequency in training")
    else:
        logger.info(f"✅ REASONABLE PROCESS COUNT: {total_new_processes} new processes")

    logger.info("Diagnostic complete.")

if __name__ == "__main__":
    main()
