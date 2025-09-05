import os
import logging
import tempfile
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional, List, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import torch
import re
import psutil
import gc
import time

from utils.env_secrets import setup_environment_secrets
setup_environment_secrets()

from config.all_config import wandb_config

logger = logging.getLogger(__name__)

# Try to import wandb, handle gracefully if not available:
import wandb
WANDB_AVAILABLE = True


def monitor_memory(func: Callable) -> Callable:
    """Decorator to monitor memory usage around external methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Skip monitoring for internal methods
        if func.__name__.startswith('_'):
            return func(self, *args, **kwargs)
            
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        try:
            result = func(self, *args, **kwargs)
            return result
        finally:
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before
            
            if abs(memory_delta) > 50:
                logger.info(f"Memory usage for {func.__name__}: {memory_before:.1f}MB → {memory_after:.1f}MB (Δ{memory_delta:+.1f}MB)")
    return wrapper


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
        # Method 1: Check if we're running under pytest
        import sys
        if 'pytest' in sys.modules or 'PYTEST_CURRENT_TEST' in os.environ:
            return True
            
        # Method 2: Check for test-related environment variables
        test_env_vars = ['TESTING', 'TEST_MODE', 'CI', 'GITHUB_ACTIONS']
        if any(os.environ.get(var, '').lower() in ['1', 'true', 'yes'] for var in test_env_vars):
            return True
            
        # Method 3: Check run name for test indicators (existing logic)
        if self.run and hasattr(self.run, 'name') and self.run.name:
            run_name = self.run.name.lower()
            test_indicators = ['test', 'wandb', 'e2e', 'integration', 'unit']
            if any(indicator in run_name for indicator in test_indicators):
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

    
    
    @monitor_memory
    @requires_wandb_enabled
    @safe_wandb_operation(default_return=False)
    def init_run(self, config: Dict[str, Any], run_name: str = wandb_config.run_name,
                 tags: Optional[List[str]] = None) -> bool:
        """
        Initialize a new wandb run.

        Args:
            config: Configuration dictionary to log
            run_name: Optional custom run name
            tags: Optional list of tags (will be merged with default tags)

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
        # Initialize wandb run
        self.run = wandb.init(**run_params)

        self.initialized = True

        logger.info(f"✅ Wandb run initialized: {self.run.name}")
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
                if hasattr(model, 'ensure_head_initialized'):
                    # prefer model API
                    try:
                        device_str = str(next(model.parameters()).device)
                    except Exception:
                        device_str = device
                    model.ensure_head_initialized(device=device_str)
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
    

    @monitor_memory
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

    @monitor_memory
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
    
    @monitor_memory
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

        # Clean up API object to prevent memory accumulation
        if self.wandb_api:
            del self.wandb_api
            self.wandb_api = None

        gc.collect()
        self.initialized = False
        logger.info("Wandb run finished and resources cleaned up")

    def _wait_for_pending_operations(self) -> None:
        """Wait for all pending async operations to complete."""
        if not self._pending_futures:
            return
            
        logger.debug(f"Waiting for {len(self._pending_futures)} pending operations...")
        for future in self._pending_futures:
            try:
                future.result(timeout=60)
            except Exception as e:
                logger.debug(f"Async operation completed with error: {e}")
        
        self._pending_futures.clear()

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
                    logger.debug(f"Could not enumerate artifact types: {e}")
                    
                    # Fallback to hardcoded common patterns if API enumeration fails
                    common_model_names = [
                        "test_collection", "test_model", "train.siamesenet_dino", 
                        "train.siamesenet", "player_model", "checkpoint"
                    ]
                    
                    for base_name in common_model_names:
                        test_name = self._get_artifact_name(base_name)
                        test_patterns.append((test_name, "model"))
                        test_patterns.append((test_name, "model_checkpoint"))
                
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

    @monitor_memory
    @requires_wandb_initialized
    @safe_wandb_operation()
    def watch_model(self, model: torch.nn.Module, log_freq: Optional[int] = None) -> None:
        """
        Watch model for gradients and parameters.
        
        Args:
            model: PyTorch model to watch
            log_freq: Frequency to log gradients
        """
        freq = log_freq or wandb_config.log_frequency
        self.run.watch(model, log_freq=freq)
        logger.info(f"Started watching model with log frequency: {freq}")

    @monitor_memory
    @requires_wandb_initialized
    @safe_wandb_operation()
    def save_checkpoint(self, epoch: int, model: Optional[torch.nn.Module] = None, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       model_state_dict: Optional[Dict[str, Any]] = None,
                       optimizer_state_dict: Optional[Dict[str, Any]] = None,
                       loss: Optional[float] = None, 
                       model_config: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Optional[str]:
        """
        Save training checkpoint to wandb artifacts for resumption purposes.
        
        Checkpoints are saved at the end of each epoch and contain:
        - Model state_dict (for model weights)  
        - Optimizer state_dict (for training state)
        - Epoch number, loss, and metadata
        
        This is separate from save_model_to_registry() which saves the final 
        production model only after successful training completion.
        
        Args:
            epoch: Current epoch number
            model: The model to save (preferred over state_dict for memory efficiency)
            optimizer: The optimizer to save (preferred over state_dict for memory efficiency)
            model_state_dict: Pre-computed model state dict (discouraged)
            optimizer_state_dict: Pre-computed optimizer state dict (discouraged)
            loss: Current loss value
            model_config: Optional model configuration metadata
            **kwargs: Additional metadata
            
        Returns:
            Artifact reference string
        """
        # Normalize inputs
        if model_state_dict is None and model is not None:
            model_state_dict = model.state_dict()
        if optimizer_state_dict is None and optimizer is not None:
            optimizer_state_dict = optimizer.state_dict()

        if model_state_dict is None:
            raise ValueError("Either model or model_state_dict must be provided")
        if optimizer_state_dict is None:
            raise ValueError("Either optimizer or optimizer_state_dict must be provided")

        # Get appropriate checkpoint name
        checkpoint_name = self._get_checkpoint_name()
        
        # Create temp file for checkpoint (ensure it's properly closed)
        with tempfile.NamedTemporaryFile(suffix=f'_epoch_{epoch}.pth', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
            # Close the file handle immediately to free resources
            tmp_file.close()
            
        # Save checkpoint data with memory optimization
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'model_config': model_config or {}
        }
        
        # Use torch.save with pickle protocol for better memory efficiency
        torch.save(checkpoint_data, checkpoint_path, pickle_protocol=4)
        
        # Clear references to large objects to help GC
        del checkpoint_data
        gc.collect()
        
        # Schedule async upload and cleanup
        if self._executor:
            future = self._executor.submit(
                self._upload_checkpoint_and_cleanup, 
                checkpoint_path, 
                checkpoint_name, 
                epoch, 
                loss
            )
            self._pending_futures.append(future)
            
            # Clean up completed futures more aggressively
            completed_futures = [f for f in self._pending_futures if f.done()]
            for future in completed_futures:
                try:
                    # Get result to ensure any exceptions are raised
                    future.result(timeout=0.1)
                except Exception as e:
                    logger.debug(f"Async operation completed with error: {e}")
                finally:
                    # Remove from list
                    if future in self._pending_futures:
                        self._pending_futures.remove(future)
            
            # Keep only non-completed futures
            self._pending_futures = [f for f in self._pending_futures if not f.done()]
            
            logger.info(f"Scheduled async checkpoint upload for epoch {epoch}")
            return f"{checkpoint_name}:latest"
        else:
            # Fallback to sync if executor unavailable
            self._upload_checkpoint_and_cleanup(checkpoint_path, checkpoint_name, epoch, loss)
            return f"{checkpoint_name}:latest"

    @monitor_memory
    def _upload_checkpoint_and_cleanup(self, checkpoint_path: str, checkpoint_name: str, epoch: int, loss: Optional[float]) -> None:
        """Upload checkpoint and clean up old versions."""
        try:
 
            self.run.log_artifact(
                artifact_or_path = checktpoin_path,
                name = checkpoint_name,
                type="model_checkpoint",
                alisases = ['latest']
            )  # Register artifact with the run
            
            logger.info(f"✅ Uploaded checkpoint for epoch {epoch}")
            
            # Clean up old checkpoints (keep only latest)
            self._cleanup_artifacts_by_type(checkpoint_name, "model_checkpoint", keep_latest=self.checkpoint_retention_count)
            
            # Periodic aggressive cleanup for long training runs
            self._checkpoint_count += 1
            if self._checkpoint_count % self._memory_cleanup_interval == 0:
                logger.info(f"🧹 Periodic memory cleanup after {self._checkpoint_count} checkpoints")
                self._force_memory_cleanup()
            
        except Exception as e:
            logger.error(f"Failed to upload checkpoint: {e}")
        finally:
            # Clean up temp file
            try:
                if os.path.exists(checkpoint_path):
                    os.unlink(checkpoint_path)
            except Exception:
                pass
            
         

    @monitor_memory
    @requires_wandb_enabled
    @safe_wandb_operation()
    def load_checkpoint(self, artifact_name: Optional[str] = None, version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Load model checkpoint from wandb.
        
        Args:
            artifact_name: Name of the checkpoint artifact (auto-detected if None)
            version: Version of the artifact to load (default: "latest")
            
        Returns:
            Checkpoint data dictionary or None if failed
        """
        if artifact_name is None:
            artifact_name = self._get_checkpoint_name()
            
        artifact_path = self._construct_artifact_path(artifact_name, version)
        
        try:
            artifact = self.wandb_api.artifact(artifact_path, type="model_checkpoint")
            artifact_dir = artifact.download()
        except Exception as e:
            logger.info(f"Checkpoint {artifact_name}:{version} not found: {e}")
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

    @monitor_memory
    def resume_training_from_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                                      artifact_name: Optional[str] = None, version: str = "latest") -> int:
        """
        Resume training from a wandb checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into  
            artifact_name: Name of the checkpoint artifact (auto-detected if None)
            version: Version to load (default: "latest")
            
        Returns:
            Starting epoch number for resuming training
        """
        checkpoint_data = self.load_checkpoint(artifact_name, version)
        
        if checkpoint_data is None:
            logger.warning("Could not load checkpoint, starting training from epoch 1")
            return 1
        
        try:
            # Load model state
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
            else:
                state_dict = checkpoint_data

            success, unexpected, missing, err = self._robust_load_state_dict(model, state_dict)
            if not success:
                logger.error(f"Failed to load model state: {err}")
                return 1
                
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys")
            if missing:
                logger.info(f"Missing keys in checkpoint: {len(missing)} keys")

            # Load optimizer state
            try:
                if isinstance(checkpoint_data, dict) and 'optimizer_state_dict' in checkpoint_data:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

            # Get starting epoch
            start_epoch = checkpoint_data.get('epoch', 0) + 1 if isinstance(checkpoint_data, dict) else 1
            logger.info(f"✅ Resumed training from epoch {start_epoch}")
            return start_epoch

        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return 1

    @monitor_memory
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
                    saved_config['embedding_dim'] = model_config_meta['embedding_dim']
                    logger.info(f"Using saved embedding_dim: {saved_config['embedding_dim']}")
        except Exception as e:
            logger.debug(f"Could not read artifact metadata: {e}")

        # Merge configurations (saved config takes precedence)
        merged_kwargs = dict(kwargs)
        merged_kwargs.update(saved_config)
        
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

    @monitor_memory
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

    @monitor_memory  
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

    @monitor_memory
    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_training_metrics(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, 
                        learning_rate: Optional[float] = None, **kwargs) -> None:
        """Log training metrics for an epoch."""
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            **kwargs
        }
        
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if learning_rate is not None:
            metrics["learning_rate"] = learning_rate
        
        self.run.log(metrics)
        logger.debug(f"Logged training metrics for epoch {epoch}")

    @monitor_memory
    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_batch_metrics(self, batch_idx: int, epoch: int, batch_loss: float, **kwargs) -> None:
        """Log batch-level metrics during training."""
        metrics = {
            "batch_loss": batch_loss,
            "epoch": epoch,
            "batch": batch_idx,
            **kwargs
        }
        self.run.log(metrics)

    @monitor_memory
    @requires_wandb_initialized
    @safe_wandb_operation()
    def update_run_config(self, config_dict: Dict[str, Any]) -> None:
        """Update the wandb run config."""
        self.run.config.update(config_dict, allow_val_change=True)
        logger.debug(f"Updated wandb run config with: {list(config_dict.keys())}")

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

    def _force_memory_cleanup(self) -> None:
        """Force aggressive memory cleanup for large artifacts during long training runs."""
        try:
            
            # Clear any pending operations that might hold large objects
            if self._pending_futures:
                # Only keep futures that are still running, remove completed ones
                self._pending_futures = [f for f in self._pending_futures if not f.done()]
            
            logger.debug("Forced memory cleanup completed")
            
        except Exception as e:
            logger.debug(f"Memory cleanup failed (non-critical): {e}")

    def force_cleanup_now(self) -> None:
        """Manually trigger memory cleanup - can be called from training code."""
        logger.info("🔧 Manual memory cleanup triggered")
        self._force_memory_cleanup()
        
        # Also clean up any completed async operations
        if self._pending_futures:
            completed_count = len([f for f in self._pending_futures if f.done()])
            if completed_count > 0:
                logger.info(f"🧹 Cleaned up {completed_count} completed async operations")

    def _monitor_wandb_processes(self) -> int:
        """Monitor and count WandB-related processes for performance testing.
        
        Returns:
            int: Number of WandB-related processes currently running
        """
        try:
            import psutil
            current_process = psutil.Process()
            wandb_process_count = 0
            
            # Count WandB-related processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'wandb' in proc.info['name'].lower():
                        wandb_process_count += 1
                    elif proc.info['cmdline']:
                        cmdline_str = ' '.join(proc.info['cmdline'])
                        if 'wandb' in cmdline_str.lower():
                            wandb_process_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            logger.debug(f"Found {wandb_process_count} WandB-related processes")
            return wandb_process_count
            
        except ImportError:
            logger.warning("psutil not available for process monitoring")
            return 0
        except Exception as e:
            logger.warning(f"Error monitoring WandB processes: {e}")
            return 0


# Global instance
wandb_logger = WandbLogger()
            