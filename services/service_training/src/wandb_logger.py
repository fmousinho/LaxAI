import gc
import logging
logger = logging.getLogger(__name__)
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import psutil
import torch

from shared_libs.config.all_config import wandb_config



# Try to import wandb, handle gracefully if not available:
import wandb

WANDB_AVAILABLE = True



def requires_wandb_enabled(func: Callable) -> Callable:
    """Decorator to check if wandb is enabled before executing method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            msg = f"Wandb not enabled or available for {func.__name__}"
            logger.error(msg)
            # Strict behavior: fail fast when wandb is not available
            raise RuntimeError(msg)
        return func(self, *args, **kwargs)
    return wrapper


def requires_wandb_initialized(func: Callable) -> Callable:
    """Decorator to check if wandb is enabled and initialized before executing method."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.enabled or not self.initialized or self.run is None:
            msg = f"Wandb logging is not enabled or initialized for {func.__name__}"
            logger.debug(msg)
            # Non-strict behavior: allow finish/other calls to be no-ops when wandb isn't initialized
            return None
        return func(self, *args, **kwargs)
    return wrapper


def safe_wandb_operation(default_return=None):
    """Decorator to safely execute wandb operations with error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to execute {func.__name__}: {e}")
                # If wandb is expected to be enabled, surface the error to fail fast
                try:
                    if wandb_config.enabled:
                        raise
                except Exception:
                    # Re-raise original exception
                    raise
                return default_return
        return wrapper
    return decorator


class StateDicts(TypedDict):
    """Type definition for checkpoint state dictionaries."""
    epoch: int
    wandb_run_id: Optional[str]
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    lr_scheduler_state_dict: Dict[str, Any]


class WandbLogger:
    """
    Refactored Weights & Biases logging integration.
    
    Key features:
    - Async by default for all heavy operations
    - Smart naming: "checkpoint"/"test-checkpoint", "model"/"test-model"  
    - Memory monitoring on all external methods
    - Lightweight cleanup without downloading artifacts
    - Configurable retention policies
    - Automatic test artifact cleanup
    """

    def __init__(self, enabled: bool = wandb_config.enabled):
        """
        Initialize the wandb logger.
        
        Args:
            enabled: Override config setting for wandb logging
        """
        self.enabled = enabled
        self.run: Any = None
        self.initialized = False
        
        # Configuration
        self.model_retention_count = getattr(wandb_config, 'model_retention_count', 3)
        self.checkpoint_retention_count = 1  # Always keep only latest checkpoint
        
        if not WANDB_AVAILABLE:
            self.enabled = False
            logger.warning("wandb package not available, disabling wandb logging")

        self.wandb_api: Any = self._login_and_get_api()
        if not self.wandb_api:
            self.enabled = False
            logger.warning("Failed to login to wandb, disabling wandb logging")

        # Single worker thread pool for async operations (preserves ordering)
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wandb-async")
        self._pending_futures: List[Future] = []
        
        # Memory management for long training runs
        self._checkpoint_count = 0
        self._memory_cleanup_interval = getattr(wandb_config, 'memory_cleanup_interval', 5)  # Configurable cleanup interval

    # --- Convenience logging proxies ---------------------------------
    # Some tests and external callers call logger.info()/warning() on the
    # WandbLogger instance. Provide thin proxy methods that forward to the
    # module-level `logger` so those calls succeed and preserve message
    # formatting semantics.
    def info(self, msg: str, *args, **kwargs) -> None:
        logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        logger.debug(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        logger.exception(msg, *args, **kwargs)
    def _get_api_key(self) -> Optional[str]:
        """Get and validate wandb API key."""
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            logger.error("WANDB_API_KEY environment variable not found")
            return None
        return api_key
    
    def _login_and_get_api(self) -> Optional[object]:
        """Login to wandb and return API object."""
        api_key = self._get_api_key()
        if not api_key:
            return None
        
        wandb.login(key=api_key)
        return wandb.Api()
    
    def _construct_artifact_path(self, artifact_name: str, version: str = "latest") -> str:
        """Construct standardized artifact path."""
        safe_name = self._sanitize_artifact_name(artifact_name)
        return f"{wandb_config.team}/{wandb_config.project}/{safe_name}:{version}"

    def _is_test_run(self) -> bool:
        """Check if current run is a test run."""
        # Method 3: Check run name for test indicators FIRST (highest priority)
        if self.run and hasattr(self.run, 'name') and self.run.name:
            run_name = self.run.name.lower()
            # More specific test indicators to avoid false positives
            test_indicators = ['test-', '-test', '_test', 'test_', 'wandb', 'e2e', 'integration', 'unit']
            if any(indicator in run_name for indicator in test_indicators):
                return True
            # If run name is production-like, don't treat as test run even under pytest
            production_indicators = ['prod', 'production', 'main', 'master', 'release', 'baseline']
            if any(indicator in run_name for indicator in production_indicators):
                return False
        
        # Method 1: Check if we're running under pytest - only if run name contains test indicators
        import sys
        if 'pytest' in sys.modules or 'PYTEST_CURRENT_TEST' in os.environ:
            # Only treat as test run if we have a run name with test indicators
            if self.run and hasattr(self.run, 'name') and self.run.name:
                run_name = self.run.name.lower()
                test_indicators = ['test-', '-test', '_test', 'test_', 'wandb', 'e2e', 'integration', 'unit']
                if any(indicator in run_name for indicator in test_indicators):
                    return True
            # If no run name or run name doesn't have test indicators, don't treat as test
            return False
            
        # Method 2: Check for test-related environment variables
        test_env_vars = ['TESTING', 'TEST_MODE', 'CI', 'GITHUB_ACTIONS']
        if any(os.environ.get(var, '').lower() in ['1', 'true', 'yes'] for var in test_env_vars):
            return True
        
        # Method 4: Check if running in test directories
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                filename = frame.f_code.co_filename
                if '/tests/' in filename or filename.endswith('test.py') or filename.startswith('test_'):
                    return True
                frame = frame.f_back
        except Exception:
            pass
        
        return False
    
    def _get_artifact_name(self, base_name: str) -> str:
        """Get artifact name with test prefix if this is a test run."""
        if self._is_test_run():
            return f"test-{base_name}"
        return base_name
    
    def _get_checkpoint_name(self) -> str:
        """Get the checkpoint artifact name with run name to prevent overwrites."""
        base_name = "checkpoint"
        
        # Include run name to make checkpoints unique per run
        if self.run and hasattr(self.run, 'name') and self.run.name:
            run_name = self.run.name
            # Sanitize run name for artifact naming
            safe_run_name = self._sanitize_artifact_name(run_name)
            base_name = f"checkpoint-{safe_run_name}"
        
        return self._get_artifact_name(base_name)

    def get_checkpoint_name(self) -> str:
        """Get the checkpoint artifact name."""
        return self._get_checkpoint_name()

    
    
    @requires_wandb_enabled
    @safe_wandb_operation(default_return=False)
    def init_run(self, config: Dict[str, Any], run_name: str = wandb_config.run_name,
                 tags: Optional[List[str]] = None, run_id: Optional[str] = None) -> bool:
        """
        Initialize a new wandb run.

        Args:
            config: Configuration dictionary to log
            run_name: Optional custom run name
            tags: Optional list of tags (will be merged with default tags)
            run_id: Optional run ID to resume a specific run

        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Merge tags
        all_tags = wandb_config.tags.copy()
        if tags:
            all_tags.extend(tags)

        run_params = {
            "project": wandb_config.project,
            "entity": wandb_config.team,
            "name": run_name or wandb_config.run_name,
            "tags": all_tags,
            "config": config,
            "reinit": "return_previous",
            "resume": "allow"
        }
        
        # If run_id is provided, use it to resume that specific run
        if run_id:
            run_params["id"] = run_id
            logger.info(f"Resuming WandB run with ID: {run_id}")
            
        # Initialize wandb run
        self.run = wandb.init(**run_params)

        self.initialized = True

        logger.info(f"✅ Wandb run initialized: {self.run.name} (ID: {self.run.id})")
        logger.info(f"   Project: {wandb_config.project}")
        logger.info(f"   Entity: {wandb_config.entity}")
        logger.info(f"   Tags: {all_tags}")

        return True

    def _robust_load_state_dict(self, model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], device: str = "cpu") -> Tuple[bool, list, list, Optional[str]]:
        """
        Robustly load a state_dict into `model`.

        Tries in order:
         - initialize lazy heads if checkpoint contains backbone._head keys
         - strict load
         - lenient load (strict=False)
         - filtered load (keep only keys that exist in model)

        Returns (success, unexpected_keys, missing_keys, error_message)
        """
        try:
            sd_keys = set(state_dict.keys())
        except Exception:
            sd_keys = set()

        # Attempt to initialize lazy head if checkpoint contains its keys
        try:
            model_backbone = getattr(model, 'backbone', None)
        except Exception:
            model_backbone = None

        if any(k.startswith('backbone._head') for k in sd_keys) and model_backbone is not None and getattr(model_backbone, '_head', None) is None:
            logger.info("Checkpoint contains backbone._head keys; attempting to initialize head before loading state_dict")
            try:
                if hasattr(model, 'ensure_head_initialized') and callable(getattr(model, 'ensure_head_initialized')):
                    # prefer model API
                    try:
                        device_str = str(next(model.parameters()).device)
                    except Exception:
                        device_str = device
                    model.ensure_head_initialized(device=device_str)  # type: ignore
                else:
                    dev = None
                    try:
                        dev = next(model.parameters()).device
                    except Exception:
                        dev = torch.device(device)
                    model.to(dev)
                    model.eval()
                    with torch.no_grad():
                        dummy = torch.zeros((1, 3, 224, 224), device=dev)
                        model(dummy)
            except Exception as e:
                logger.warning(f"Could not initialize model head before loading state_dict: {e}")

        model_state = model.state_dict()

        # Try strict load first
        try:
            model.load_state_dict(state_dict)
            try:
                model.to(device)
            except Exception:
                pass
            return True, [], [], None
        except RuntimeError as e_strict:
            logger.warning(f"Strict state_dict load failed: {e_strict}")

        # Try lenient load
        try:
            model.load_state_dict(state_dict, strict=False)
            try:
                model.to(device)
            except Exception:
                pass
            sd_keys = set(state_dict.keys())
            unexpected = [k for k in sd_keys if k not in model_state]
            missing = [k for k in model_state.keys() if k not in sd_keys]
            return True, unexpected, missing, None
        except Exception as e_lenient:
            logger.warning(f"Lenient state_dict load failed: {e_lenient}")

        # Final attempt: filter keys to only those present in the model
        try:
            filtered = {k: v for k, v in state_dict.items() if k in model_state}
            model.load_state_dict(filtered)
            try:
                model.to(device)
            except Exception:
                pass
            missing = [k for k in model_state.keys() if k not in filtered]
            return True, [], missing, None
        except Exception as e_final:
            err = f"Failed to load state_dict after all attempts: {e_final}"
            logger.error(err)
            return False, [], [], err
    

    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log summary metrics to wandb.
        
        Args:
            summary: Dictionary of summary metrics to log
        """
        self.run.summary.update(summary)
        logger.info(f"Logged summary metrics: {list(summary.keys())}")

    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        self.run.log(metrics, step=step)
    
    @requires_wandb_initialized
    @safe_wandb_operation()
    def finish(self) -> None:
        """Finish the current wandb run and cleanup."""
        # Wait for any pending async operations
        self._wait_for_pending_operations()
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
            
        # Auto-cleanup test artifacts if this is a test run
        if self.run and self._is_test_run():
            try:
                logger.info("🧹 Auto-cleaning test artifacts before finish")
                self._cleanup_test_artifacts()
            except Exception as e:
                logger.debug(f"Auto-cleanup failed (non-critical): {e}")
                
        if self.run:
            self.run.finish()

        logger.info("Wandb run finished and resources cleaned up")

    def _wait_for_pending_operations(self) -> None:
        """Wait for all pending async operations to complete."""
        if not self._pending_futures:
            return
            
        logger.debug(f"Waiting for {len(self._pending_futures)} pending operations...")
        critical_errors = []
        
        for future in self._pending_futures:
            try:
                # Get result to ensure any exceptions are raised
                future.result(timeout=60)
            except (NameError, AttributeError, TypeError) as e:
                # Critical errors indicate bugs - collect and re-raise
                logger.error(f"Critical error in async operation: {e}")
                critical_errors.append(e)
            except Exception as e:
                logger.debug(f"Async operation completed with error: {e}")
        
        self._pending_futures.clear()
        
        # Re-raise critical errors after processing all futures
        if critical_errors:
            raise critical_errors[0]

    def _cleanup_test_artifacts(self) -> None:
        """Clean up all artifacts created by test runs."""
        if not self.run:
            return
            
        try:
            # Get all artifacts created by this run using API
            artifacts_to_delete = []
            
            try:
                # Use API to get artifacts from this specific run
                api_run = self.wandb_api.run(f"{self.run.entity}/{self.run.project}/{self.run.id}")
                
                # Get output artifacts (ones created by this run)
                output_artifacts = api_run.output_artifacts()
                artifacts_to_delete = list(output_artifacts)
                logger.debug(f"Found {len(artifacts_to_delete)} output artifacts to clean up")
                
            except Exception as e:
                logger.debug(f"Could not get output artifacts via API: {e}")
            
            # Fallback: try to clean up common test artifact patterns
            if not artifacts_to_delete:
                logger.debug("No output artifacts found, trying pattern-based cleanup")
                
                # Try to find test artifacts by name patterns
                test_patterns = []
                
                # Add checkpoint patterns
                checkpoint_name = self._get_checkpoint_name()
                test_patterns.append((checkpoint_name, "model_checkpoint"))
                
                # Add model patterns - be more comprehensive about test model patterns
                # Check if any artifacts match test patterns in the project
                try:
                    # Get all model artifact types in the project
                    model_artifact_types = []
                    try:
                        model_artifact_types.append(self.wandb_api.artifact_type("model", project=wandb_config.project))
                    except Exception:
                        pass
                    
                    try:
                        model_artifact_types.append(self.wandb_api.artifact_type("model_checkpoint", project=wandb_config.project))
                    except Exception:
                        pass
                    
                    # Find collections that start with "test-"
                    for artifact_type_obj in model_artifact_types:
                        try:
                            collections = artifact_type_obj.collections()
                            for collection in collections:
                                if collection.name.startswith("test-"):
                                    test_patterns.append((collection.name, artifact_type_obj.name))
                                    logger.debug(f"Found test artifact pattern: {collection.name} (type: {artifact_type_obj.name})")
                        except Exception as e:
                            logger.debug(f"Could not get collections for {artifact_type_obj.name}: {e}")
                            
                except Exception as e:
                    logger.error(f"Could not enumerate artifact types: {e}")
                
                # Remove duplicates
                test_patterns = list(set(test_patterns))
                
                # Try to clean up each pattern
                for artifact_name, artifact_type in test_patterns:
                    try:
                        # Check if collection exists first
                        artifact_collection = self.wandb_api.artifact_type(artifact_type, project=wandb_config.project).collection(artifact_name)
                        if artifact_collection:
                            artifacts_to_delete.append(type('MockArtifact', (), {'name': artifact_name, 'type': artifact_type})())
                            logger.debug(f"Added {artifact_name} ({artifact_type}) to cleanup list")
                    except Exception:
                        # Collection doesn't exist, skip
                        continue
            
            # Delete all found artifacts
            deleted_count = 0
            for artifact in artifacts_to_delete:
                try:
                    # Use lightweight deletion without downloading
                    self._lightweight_artifact_delete(artifact.name, artifact.type)
                    deleted_count += 1
                except Exception as e:
                    logger.debug(f"Failed to delete artifact {artifact.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"✅ Cleaned up {deleted_count} test artifacts")
            else:
                logger.info("ℹ️  No test artifacts found to clean up")
                
        except Exception as e:
            logger.warning(f"Test artifact cleanup failed: {e}")

    def cleanup_test_artifacts(self, force_cleanup_all: bool = False) -> None:
        """Public method for test artifact cleanup (backwards compatibility)."""
        self._cleanup_test_artifacts()

    def _lightweight_artifact_delete(self, artifact_name: str, artifact_type: str) -> None:
        """Delete artifact without downloading it first."""
        try:
            # Try to get the collection first
            artifact_collection = self.wandb_api.artifact_type(artifact_type, project=wandb_config.project).collection(artifact_name)
            
            # Try to delete the entire collection
            try:
                artifact_collection.delete()
                logger.debug(f"Deleted artifact collection: {artifact_name}")
                return
            except Exception as e:
                logger.debug(f"Collection delete failed for {artifact_name}, trying per-version: {e}")
            
            # Fallback to per-version deletion
            artifacts = list(artifact_collection.artifacts())
            deleted_count = 0
            for artifact in artifacts:
                try:
                    artifact.delete()
                    deleted_count += 1
                except Exception as e:
                    logger.debug(f"Failed to delete version {artifact.version} of {artifact_name}: {e}")
            
            if deleted_count > 0:
                logger.debug(f"Deleted {deleted_count} versions of artifact: {artifact_name}")
            else:
                raise Exception(f"No versions could be deleted for {artifact_name}")
                
        except Exception as e:
            # Check if it's just that the artifact doesn't exist
            if "404" in str(e) or "not found" in str(e).lower():
                logger.debug(f"Artifact {artifact_name} not found (already deleted or never existed)")
            else:
                logger.debug(f"Could not delete artifact {artifact_name}: {e}")
                raise



    @requires_wandb_initialized
    @safe_wandb_operation()
    def save_checkpoint(self, state_dicts: StateDicts, epoch: int, task_id: str, wandb_run_id: Optional[str] = None) -> Optional[str]:
        """
        Save training checkpoint to wandb artifacts for resumption purposes.
    
        This is separate from save_model_to_registry() which saves the final 
        production model only after successful training completion.
        
        Args:
            state_dicts: Dictionary containing:
                - model_state_dict: Model's state dict
                - optimizer_state_dict: Optimizer's state dict
                - lr_scheduler_state_dict: LR scheduler's state dict
                - epoch: Current epoch number
                - wandb_run_id: Optional WandB run ID to persist
            epoch: Current epoch number (for filename)
            task_id: Training task identifier (not used here)
            wandb_run_id: Optional WandB run ID to persist (if not in state_dicts)
            
        Returns:
            Artifact reference string
        """
        # Validate checkpoint data
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'lr_scheduler_state_dict']
        missing_keys = [key for key in required_keys if key not in state_dicts]
        if missing_keys:
            raise ValueError(f"checkpoint_data missing required keys: {missing_keys}")

        # Add wandb_run_id to state_dicts if provided
        if wandb_run_id and 'wandb_run_id' not in state_dicts:
            state_dicts['wandb_run_id'] = wandb_run_id

        # Check epoch consistency
        if 'lr_scheduler_state_dict' in state_dicts:
            scheduler_state = state_dicts['lr_scheduler_state_dict']
            if scheduler_state and 'last_epoch' in scheduler_state:
                scheduler_epoch = scheduler_state['last_epoch']
                # Note: scheduler epoch is often 0-indexed or 1-indexed depending on implementation
                # We just check if they are different and log it
                if scheduler_epoch!= epoch:
                    logger.error(f"Epoch mismatch: Argument epoch={epoch}, Scheduler epoch={scheduler_epoch}")

        # Get appropriate checkpoint name
        checkpoint_name = self._get_checkpoint_name()
        
        # Create temp file for checkpoint (ensure it's properly closed)
        with tempfile.NamedTemporaryFile(suffix=f'_epoch_{epoch}.pth', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
            # Close the file handle immediately to free resources
            tmp_file.close()
            
        # Use torch.save with pickle protocol for better memory efficiency
        torch.save(state_dicts, checkpoint_path, pickle_protocol=4)
        
        # Schedule async upload and cleanup
        if self._executor:
            future = self._executor.submit(
                self._trigger_checkpoint_upload_and_cleanup,
                checkpoint_path,
                checkpoint_name,
            )
            self._pending_futures.append(future)

            # Clean up completed futures more aggressively
            self._cleanup_completed_futures()

            return f"{checkpoint_name}:latest"
        else:
            # Fallback to sync if executor unavailable
            self._trigger_checkpoint_upload_and_cleanup(
                checkpoint_path, checkpoint_name
            )
            return f"{checkpoint_name}:latest"

    def _cleanup_completed_futures(self):
        """Iterate through pending futures and clean up any that are done."""
        completed_futures = [f for f in self._pending_futures if f.done()]
        for future in completed_futures:
            try:
                # Get result to ensure any exceptions are raised
                future.result(timeout=0.1)
            except (NameError, AttributeError, TypeError) as e:
                # Critical errors indicate bugs - re-raise these
                logger.error(f"Critical error in async operation: {e}")
                raise
            except Exception as e:
                logger.debug(f"Async operation completed with error: {e}")
            finally:
                # Remove from list
                if future in self._pending_futures:
                    self._pending_futures.remove(future)
        # Keep only non-completed futures
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def _trigger_checkpoint_upload_and_cleanup(
        self, checkpoint_path: str, checkpoint_name: str
    ) -> None:
        """
        Orchestrates checkpoint upload via subprocess and then cleans up old artifacts.
        This method is the target for the ThreadPoolExecutor.
        """
        try:
            # Step 1: Upload the artifact in a separate process to isolate memory.
            self._launch_checkpoint_subprocess(checkpoint_path, checkpoint_name)

            # Step 2: Clean up old checkpoints in the main process.
            self._cleanup_artifacts_by_type(
                checkpoint_name, "model_checkpoint", keep_latest=self.checkpoint_retention_count
            )

        except (NameError, AttributeError, TypeError) as e:
            logger.error(f"Critical error during checkpoint orchestration (likely bug): {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to orchestrate checkpoint upload and cleanup: {e}")
        finally:
            # Always clean up the temp file, even if subprocess failed
            try:
                if os.path.exists(checkpoint_path):
                    os.unlink(checkpoint_path)
                    logger.debug(f"Cleaned up temp file: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {checkpoint_path}: {e}")

    def _launch_checkpoint_subprocess(self, checkpoint_path: str, checkpoint_name: str) -> None:
        """
        Launches a separate process to upload a checkpoint artifact to wandb.
        This isolates the memory usage of the upload from the main training process.
        """
        if not self.run or not self.run.id:
            raise RuntimeError("Wandb run is not initialized or has no ID.")

        run_id = self.run.id
        script_path = os.path.abspath(__file__)
        python_executable = sys.executable

        command = [
            python_executable,
            script_path,
            "--upload-checkpoint",
            "--run-id", run_id,
            "--checkpoint-path", checkpoint_path,
            "--checkpoint-name", checkpoint_name,
            "--project", wandb_config.project,
            "--entity", wandb_config.team,
        ]

        logger.info(f"Launching subprocess for checkpoint upload: {' '.join(command)}")

        try:
            # We use subprocess.run and capture output for better error diagnosis.
            # This call is blocking within its thread, which is intended.
            # Make a copy of the current environment and ensure the repository root
            # is on PYTHONPATH so the subprocess can import local packages like
            # `shared_libs`. Also set cwd to the repo root for predictable imports.
            child_env = os.environ.copy()
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            existing_py = child_env.get('PYTHONPATH', '')
            # Prepend repo_root if not already present
            if repo_root not in existing_py.split(os.pathsep):
                child_env['PYTHONPATH'] = os.pathsep.join([repo_root, existing_py]) if existing_py else repo_root

            result = subprocess.run(
                command,
                check=True,  # Raises CalledProcessError if the command returns a non-zero exit code.
                capture_output=True,
                text=True,
                env=child_env,
                cwd=repo_root,
            )
            logger.debug(f"Subprocess stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Subprocess stderr: {result.stderr}")
            
            # Also log subprocess output at INFO level for Google Cloud Logger
            if result.stdout.strip():
                logger.info(f"Subprocess output: {result.stdout.strip()}")

        except subprocess.CalledProcessError as e:
            # Log detailed error information from the subprocess
            error_message = (
                f"Subprocess for checkpoint upload failed with exit code {e.returncode}.\n"
                f"  Command: {' '.join(e.cmd)}\n"
                f"  Stdout: {e.stdout}\n"
                f"  Stderr: {e.stderr}"
            )
            logger.error(error_message)
            # Don't re-raise the exception to avoid breaking training
            # Just log the error and continue
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred while launching subprocess: {e}")
            # Don't re-raise to avoid breaking training
            return

   
         
    @requires_wandb_enabled
    @safe_wandb_operation()
    def load_checkpoint(self, artifact_name: Optional[str] = None, version: str = "latest", run_name: Optional[str] = None) -> Optional[StateDicts]:
        """
        Load model checkpoint from wandb.
        
        Args:
            artifact_name: Name of the checkpoint artifact (auto-detected if None)
            version: Version of the artifact to load (default: "latest")
            run_name: Run name to use for checkpoint name generation (uses self.run.name if None)
            
        Returns:
            Checkpoint data dictionary or None if failed
        """
        if artifact_name is None:
            # Use provided run_name or fall back to self.run.name
            if run_name:
                safe_run_name = self._sanitize_artifact_name(run_name)
                artifact_name = f"checkpoint-{safe_run_name}"
            else:
                artifact_name = self._get_checkpoint_name()
            logger.info(f"Auto-detected checkpoint name: {artifact_name}")
            
        artifact_path = self._construct_artifact_path(artifact_name, version)
        logger.info(f"Constructed artifact path: {artifact_path}")
        
        # Try to load with sanitized name first
        artifact_dir = None
        try:
            logger.info(f"Attempting to download artifact from WandB...")
            artifact = self.wandb_api.artifact(artifact_path, type="model_checkpoint")
            artifact_dir = artifact.download()
            logger.info(f"✅ Successfully downloaded artifact to: {artifact_dir}")
        except Exception as e:
            logger.warning(f"❌ Checkpoint {artifact_path} not found: {e}")
            
            # Fallback: try with original run name (no sanitization) if we have a run_name
            if run_name:
                original_name = f"checkpoint-{run_name}"
                fallback_path = self._construct_artifact_path(original_name, version)
                logger.info(f"Trying fallback with original run name: {fallback_path}")
                try:
                    artifact = self.wandb_api.artifact(fallback_path, type="model_checkpoint")
                    artifact_dir = artifact.download()
                    logger.info(f"✅ Successfully downloaded artifact using fallback to: {artifact_dir}")
                except Exception as e2:
                    logger.warning(f"❌ Fallback also failed: {e2}")
                    return None
            elif self.run and hasattr(self.run, 'name') and self.run.name:
                # Fallback using self.run.name if available
                original_name = f"checkpoint-{self.run.name}"
                fallback_path = self._construct_artifact_path(original_name, version)
                logger.info(f"Trying fallback with self.run.name: {fallback_path}")
                try:
                    artifact = self.wandb_api.artifact(fallback_path, type="model_checkpoint")
                    artifact_dir = artifact.download()
                    logger.info(f"✅ Successfully downloaded artifact using fallback to: {artifact_dir}")
                except Exception as e2:
                    logger.warning(f"❌ Fallback also failed: {e2}")
                    return None
            else:
                return None

        # Find checkpoint file
        checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
        
        if not checkpoint_files:
            logger.error(f"No checkpoint files found in artifact {artifact_name}:{version}")
            return None
        
        # Load the checkpoint
        checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        logger.info(f"✅ Loaded checkpoint: {artifact_name}:{version} (epoch {checkpoint_data.get('epoch', 'unknown')})")
        return checkpoint_data


    @requires_wandb_enabled
    @safe_wandb_operation()
    def load_model_from_registry(self, model_class, collection_name: str, 
                                alias: str = "latest", device: str = "cpu", **kwargs) -> Optional[torch.nn.Module]:
        """
        Load model from wandb model registry.
        
        Args:
            model_class: The model class to instantiate
            collection_name: Name of the model collection in wandb registry
            alias: Model version alias (e.g., "latest", "best", "v1")
            device: Device to load the model on
            **kwargs: Additional model initialization parameters
            
        Returns:
            Loaded model instance or None if loading failed
        """
        # Use appropriate model name based on test status
        model_name = self._get_artifact_name(collection_name)
        artifact_path = self._construct_artifact_path(model_name, alias)
        
        try:
            model_artifact = self.wandb_api.artifact(artifact_path, type="model")
            model_dir = model_artifact.download()
        except Exception as e:
            logger.warning(f"Could not retrieve model artifact '{artifact_path}': {e}")
            return None

        # Find model file
        model_file_name = None
        for file in model_artifact.files():
            if file.name.endswith('.pth'):
                model_file_name = file.name
                break

        if not model_file_name:
            logger.error(f"No .pth file found in artifact: {artifact_path}")
            return None

        model_path = os.path.join(model_dir, model_file_name)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None

        # Read saved configuration from metadata
        saved_config = {}
        try:
            if hasattr(model_artifact, 'metadata') and model_artifact.metadata:
                model_config_meta = model_artifact.metadata.get('model_config', {})
                if 'embedding_dim' in model_config_meta:
                    # Ensure embedding_dim is an integer
                    saved_config['embedding_dim'] = int(model_config_meta['embedding_dim'])
        except Exception as e:
            logger.debug(f"Could not read artifact metadata: {e}")

        # Merge configurations (saved config takes precedence)
        merged_kwargs = dict(kwargs)
        merged_kwargs.update(saved_config)
        
        logger.info(f"DEBUG: Initializing model class {model_class.__name__} with kwargs: {merged_kwargs}")

        # Initialize model
        model = model_class(**merged_kwargs)

        # Load model weights
        raw = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            state_dict = raw['model_state_dict']
        else:
            state_dict = raw

        success, unexpected, missing, err = self._robust_load_state_dict(model, state_dict, device=device)
        if success:
            if unexpected:
                logger.warning(f"Model had unexpected keys: {len(unexpected)} keys")
            if missing:
                logger.info(f"Model had missing keys: {len(missing)} keys")
            logger.info(f"✅ Loaded model from registry: {collection_name}:{alias}")
            return model
        else:
            logger.error(f"Failed to load model: {err}")
            return None

    @requires_wandb_initialized
    @safe_wandb_operation()
    def save_model_to_registry(self, model: torch.nn.Module, collection_name: str, 
                            alias: str = "latest", file_name: str = "model.pth", 
                            metadata: Optional[Dict[str, Any]] = None, 
                            tags: Optional[List[str]] = None) -> None:
        """
        Save final production model to wandb model registry.
        
        This should ONLY be called after successful completion of training.
        It saves the final trained model for production use, not for training resumption.
        
        Key differences from save_checkpoint():
        - Saves only model state_dict (no optimizer state)
        - Intended for production deployment 
        - Called once at end of successful training
        - Does NOT clean up checkpoints (they serve different purposes)

        Args:
            model: PyTorch model to save
            collection_name: Name for the model collection
            alias: Model version alias (e.g., "latest", "best", "v1")
            file_name: Name of the model file to save
            metadata: Additional metadata to include
            tags: Optional list of tags to add to the artifact
        """
        # Use appropriate model name based on test status
        is_test = self._is_test_run()
        model_name = self._get_artifact_name(collection_name)
        
        # Critical safety check: Always force test prefix if we detect ANY test environment
        import sys
        forced_test_prefix = False
        
        # Multiple redundant checks to prevent production model overwrites
        test_indicators = [
            'pytest' in sys.modules,
            'PYTEST_CURRENT_TEST' in os.environ,
            hasattr(sys, '_getframe') and any('/tests/' in f.f_code.co_filename 
                                            for f in (sys._getframe(i) for i in range(10)) 
                                            if f is not None),
            collection_name and any(indicator in collection_name.lower() 
                                  for indicator in ['test', 'dummy', 'mock', 'fake'])
        ]
        
        if any(test_indicators) and not model_name.startswith('test-'):
            model_name = f"test-{model_name}"
            forced_test_prefix = True
            logger.error(f"SAFETY OVERRIDE: Detected test environment, forcing test prefix! Model: '{collection_name}' -> '{model_name}'")
        
        # Log critical information about test detection
        if is_test or forced_test_prefix:
            logger.warning(f"TEST RUN DETECTED: Saving model '{collection_name}' as '{model_name}' (test-prefixed)")
        else:
            logger.info(f"Production model save: '{collection_name}' as '{model_name}'")
            # Final safety check - if this looks like a test model name, warn
            if any(indicator in collection_name.lower() for indicator in ['test', 'dummy', 'e2e', 'integration']):
                logger.warning(f"Model name '{collection_name}' looks like a test but no test environment detected!")
        
        # Auto-detect model metadata
        auto_meta = self._extract_model_metadata(model)
        
        # Merge metadata
        merged_meta = {} if metadata is None else dict(metadata)
        merged_meta.setdefault('model_config', {})
        merged_meta['model_config'].update(auto_meta)
        merged_meta['is_test'] = self._is_test_run()

        # Create artifact
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=merged_meta
        )

        # Save model to temp file and add to artifact
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            torch.save(model.state_dict(), temp_path)
            artifact.add_file(temp_path, name=file_name)

            # Log artifact
            logged_artifact = self.run.log_artifact(artifact, type="model", tags=tags)
            
            # Link to registry
            target_path = f"wandb-registry-model/{model_name}"
            try:
                self.run.link_artifact(logged_artifact, target_path=target_path)
            except Exception as e:
                logger.warning(f"Failed to link artifact to registry: {e}")

            logger.info(f"✅ Model saved to registry: {model_name}")

            # Schedule async cleanup of old model versions only
            if self._executor:
                cleanup_future = self._executor.submit(
                    self._cleanup_artifacts_by_type,
                    model_name,
                    "model", 
                    self.model_retention_count
                )
                self._pending_futures.append(cleanup_future)
                
                # Note: We do NOT clean up checkpoints when a final model is saved.
                # Checkpoints serve a different purpose (training resumption) and should
                # be retained for debugging and recovery purposes. They are managed
                # separately via checkpoint_retention_count during checkpoint saves.

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def _extract_model_metadata(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Extract metadata from model for artifact storage."""
        auto_meta = {}
        try:
            # Common model attributes
            auto_meta['embedding_dim'] = getattr(model, 'embedding_dim', None)
            auto_meta['dropout'] = getattr(model, 'dropout', getattr(model, 'dropout_rate', None))

            # Try to detect backbone head configuration
            head_info = {}
            backbone = getattr(model, 'backbone', None)
            if backbone is not None and getattr(backbone, '_head', None) is not None:
                head = backbone._head
                for name, mod in head.named_modules():
                    if isinstance(mod, torch.nn.Linear):
                        head_info[f'linear_{name}'] = {
                            'in_features': mod.in_features, 
                            'out_features': mod.out_features
                        }
                if head_info:
                    auto_meta['head'] = head_info
        except Exception as e:
            logger.debug(f"Failed to extract model metadata: {e}")

        return auto_meta

    def _cleanup_artifacts_by_type(self, collection_name: str, artifact_type: str, keep_latest: int) -> None:
        """Clean up old artifacts of a specific type, keeping only the latest N versions."""
        try:
            # Add null check for wandb_api
            if self.wandb_api is None:
                logger.warning(f"WandB API not available, skipping cleanup of {artifact_type} artifacts for {collection_name}")
                return

            artifact_type_obj = self.wandb_api.artifact_type(artifact_type, project=wandb_config.project)
            if artifact_type_obj is None:
                logger.warning(f"Could not get artifact type '{artifact_type}' from WandB API, skipping cleanup for {collection_name}")
                return

            artifact_collection = artifact_type_obj.collection(collection_name)
            if artifact_collection is None:
                logger.debug(f"Collection '{collection_name}' not found, nothing to cleanup")
                return

            # Fast path for deletion of all versions
            if keep_latest == 0:
                try:
                    artifact_collection.delete()
                    logger.info(f"🗑️  Deleted entire collection: {collection_name}")
                    return
                except Exception as e:
                    logger.debug(f"Collection delete failed, falling back to per-version: {e}")

            # Get all artifacts, sorted by creation time (newest first)
            artifacts = list(artifact_collection.artifacts())
            logger.debug(f"Found {len(artifacts)} {artifact_type} artifacts in collection {collection_name}")

            if len(artifacts) <= keep_latest:
                logger.debug(f"Only {len(artifacts)} versions exist for {collection_name}, keeping {keep_latest} - no cleanup needed")
                return

            artifacts.sort(key=lambda x: x.created_at, reverse=True)

            # Identify versions to delete (keep latest N)
            to_delete = artifacts[keep_latest:]
            logger.debug(f"Identified {len(to_delete)} {artifact_type} artifacts for potential deletion in {collection_name}")            # Skip versions marked as "do not delete"
            filtered_to_delete = []
            for artifact in to_delete:
                do_not_delete = False

                # Check aliases
                try:
                    if hasattr(artifact, 'aliases'):
                        # Use the config value with proper fallback to match all_config.py
                        protected_aliases = getattr(wandb_config, 'model_tags_to_skip_deletion', ["do_not_delete"])
                        if any(alias in protected_aliases for alias in artifact.aliases):
                            do_not_delete = True
                            logger.debug(f"Skipping deletion of {artifact.name} due to protected alias: {artifact.aliases}")
                except Exception:
                    pass

                # Check metadata
                try:
                    if artifact.metadata and artifact.metadata.get("do_not_delete", False):
                        do_not_delete = True
                        logger.debug(f"Skipping deletion of {artifact.name} due to metadata protection")
                except Exception:
                    pass

                if not do_not_delete:
                    filtered_to_delete.append(artifact)

            # Delete old versions
            deleted_count = 0
            for artifact in filtered_to_delete:
                try:
                    artifact.delete()
                    deleted_count += 1
                    logger.debug(f"🗑️  Deleted {artifact_type}: {artifact.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {artifact.name}: {e}")

            if deleted_count > 0:
                logger.info(f"🗑️  Cleaned up {deleted_count} old {artifact_type} artifacts for {collection_name}")
            else:
                logger.debug(f"No {artifact_type} artifacts were deleted for {collection_name} (all were protected or failed)")

        except AttributeError as e:
            if "'NoneType' object has no attribute" in str(e):
                logger.error(f"WandB API returned None object during cleanup of {collection_name}: {e}")
                logger.error("This usually indicates WandB API initialization issues or network connectivity problems")
            else:
                logger.error(f"Attribute error during artifact cleanup for {collection_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to cleanup {artifact_type} artifacts for {collection_name}: {e}")

    @requires_wandb_initialized
    @safe_wandb_operation()
    def cleanup_checkpoints(self, keep_latest: int = 2) -> None:
        """
        Explicitly clean up checkpoint artifacts, keeping only the latest N versions.
        
        This method allows explicit checkpoint cleanup separate from model saving.
        Useful for maintenance or when checkpoint retention needs to be adjusted.
        
        Args:
            keep_latest: Number of latest checkpoint versions to keep (default: 2)
        """
        if self._executor:
            checkpoint_name = self._get_checkpoint_name()
            cleanup_future = self._executor.submit(
                self._cleanup_artifacts_by_type,
                checkpoint_name,
                "model_checkpoint",
                keep_latest
            )
            self._pending_futures.append(cleanup_future)
            logger.info(f"🧹 Scheduled cleanup of old checkpoints, keeping latest {keep_latest} versions")
        else:
            logger.warning("No executor available for checkpoint cleanup")


    def _sanitize_artifact_name(self, name: Optional[str]) -> str:
        """Sanitize artifact names for wandb compatibility."""
        if not name:
            return "artifact"
        # Replace invalid chars with underscores
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))
        safe = safe.strip('_.-')
        if not safe:
            return "artifact"
        if safe != name:
            logger.debug(f"Sanitized artifact name: '{name}' -> '{safe}'")
        return safe



# Global instance
wandb_logger = WandbLogger()


# --- Subprocess Script Logic ---
# This block allows the file to be executed as a standalone script
# for memory-isolated artifact uploads.

def _upload_checkpoint_in_subprocess(
    run_id: str,
    checkpoint_path: str,
    checkpoint_name: str,
    project: str,
    entity: str,
):
    """
    This function runs in a separate process to upload a checkpoint.
    It initializes a wandb run, uploads the artifact, and finishes.
    """
    run = None
    try:
        # Set up logging for the subprocess - use the same format as main process
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Subprocess started for run_id: {run_id}")

        # Check if the checkpoint file exists before proceeding
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found at: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

        # Create a temporary run for artifact upload to avoid conflicts
        # We can't resume the main run from subprocess as it's already active
        temp_run_name = f"temp_checkpoint_upload_{uuid.uuid4().hex[:8]}"
        logger.debug(f"Creating temporary wandb run for artifact upload: {temp_run_name}")
        
        run = wandb.init(
            project=project,
            entity=entity,
            name=temp_run_name,
            job_type="checkpoint_upload",
            tags=["temp", "checkpoint"],
            settings=wandb.Settings(init_timeout=300)  # 5 minute timeout should be sufficient
        )

        if run is None:
            raise ConnectionError(f"Failed to create temporary wandb run for checkpoint upload.")

        logger.debug(f"Successfully created temporary run {run.name} (ID: {run.id}) for artifact upload.")

        # Create and log the artifact to the temporary run
        artifact = wandb.Artifact(
            name=checkpoint_name,
            type="model_checkpoint",
            description=f"Checkpoint uploaded from run {run_id}"
        )
        artifact.add_file(checkpoint_path)

        # Upload the artifact
        run.log_artifact(artifact, aliases=["latest"])
        
        # Note: The artifact is now available in the project and can be accessed by name
        # from any run in the same project, including the original run_id

        logger.info(f"✅ Successfully uploaded artifact '{checkpoint_name}' in subprocess.")
        
        # Clean up wandb artifact cache after successful upload
        try:
            import subprocess
            logger.debug("Running wandb artifact cache cleanup...")
            result = subprocess.run(
                ["wandb", "artifact", "cache", "cleanup", "--remove-temp", "1GB"],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            if result.returncode == 0:
                logger.info("✅ Wandb artifact cache cleanup completed")
                if result.stdout.strip():
                    logger.debug(f"Cache cleanup output: {result.stdout.strip()}")
            else:
                logger.warning(f"Wandb cache cleanup failed with exit code {result.returncode}")
                if result.stderr:
                    logger.warning(f"Cache cleanup stderr: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            logger.warning("Wandb cache cleanup timed out")
        except FileNotFoundError:
            logger.warning("wandb CLI not found, skipping cache cleanup")
        except Exception as e:
            logger.warning(f"Failed to run wandb cache cleanup: {e}")

    except Exception as e:
        logger.error(f"Error in checkpoint upload subprocess: {e}", exc_info=True)
        # Re-raise to ensure the parent process sees a non-zero exit code
        raise
    finally:
        # Ensure the run is finished to stop the subprocess
        if run:
            run_id_to_delete = run.id
            run_path = f"{entity}/{project}/{run_id_to_delete}"
            run.finish()
            
            # Delete the temporary run but keep the artifacts
            try:
                logger.info(f"Attempting to delete temporary run: {run_path}")
                api = wandb.Api()
                run_obj = api.run(run_path)
                run_obj.delete(delete_artifacts=False)
                logger.info(f"✅ Successfully deleted temporary run: {run_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary run {run_path}: {e}")
        # Clean up the temporary file
        if os.path.exists(checkpoint_path):
            try:
                os.unlink(checkpoint_path)
            except OSError as e:
                logger.error(f"Error removing temp file {checkpoint_path}: {e}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="WandbLogger utility script.")
    parser.add_argument("--upload-checkpoint", action="store_true", help="Run the checkpoint upload task.")
    parser.add_argument("--run-id", type=str, help="Wandb run ID to resume.")
    parser.add_argument("--checkpoint-path", type=str, help="Path to the checkpoint file to upload.")
    parser.add_argument("--checkpoint-name", type=str, help="Name of the checkpoint artifact.")
    parser.add_argument("--project", type=str, help="Wandb project name.")
    parser.add_argument("--entity", type=str, help="Wandb entity (team) name.")

    args = parser.parse_args()

    if args.upload_checkpoint:
        if not all([args.run_id, args.checkpoint_path, args.checkpoint_name, args.project, args.entity]):
            logger.error("Error: --run-id, --checkpoint-path, --checkpoint-name, --project, and --entity are required for --upload-checkpoint.")
            sys.exit(1)
        
        try:
            _upload_checkpoint_in_subprocess(
                run_id=args.run_id,
                checkpoint_path=args.checkpoint_path,
                checkpoint_name=args.checkpoint_name,
                project=args.project,
                entity=args.entity,
            )
            sys.exit(0)
        except Exception as e:
            # The error is already logged in the function, just exit with failure.
            sys.exit(1)
