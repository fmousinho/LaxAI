import os
import logging
import tempfile
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional, List, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import torch
import re
import psutil
import signal
import time

from utils.env_secrets import setup_environment_secrets
setup_environment_secrets()

from config.all_config import wandb_config

logger = logging.getLogger(__name__)

# Try to import wandb, handle gracefully if not available:
import wandb
WANDB_AVAILABLE = True


CHECKPOINT_NAME = "checkpoint"


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
    Utility class for Weights & Biases logging integration.
    Handles experiment tracking, metrics logging, and model artifacts.
    """

    def __init__(self, enabled: bool = wandb_config.enabled):
        """
        Initialize the wandb logger.
        
        Args:
            enabled: Override config setting for wandb logging
        """
        self.enabled = enabled
        # Use Any to avoid static analyzer complaints about wandb runtime types
        self.run: Any = None
        self.initialized = False
        self.wandb_processes = []  # Track WandB background processes

        if not WANDB_AVAILABLE:
            self.enabled = False
            logger.warning("wandb package not available, disabling wandb logging")

        self.wandb_api: Any = self._login_and_get_api()
        if not self.wandb_api:
            self.enabled = False
            logger.warning("Failed to login to wandb, disabling wandb logging")

        # Thread pool for non-blocking artifact uploads / cleanup
        # Single worker preserves ordering and avoids excess memory usage.
        self._io_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1)
        self._pending_checkpoint_futures: List[Future] = []

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
        # Ensure environment is properly loaded first
        
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

    def _monitor_wandb_processes(self) -> int:
        """
        Monitor and return the count of WandB-related background processes.
        
        Returns:
            Number of WandB background processes found
        """
        try:
            current_process = psutil.Process()
            all_processes = []
            
            # Get all child processes
            try:
                all_processes = current_process.children(recursive=True)
            except psutil.NoSuchProcess:
                pass
            
            # Also check all processes for wandb-related ones
            wandb_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'wandb' in proc.info['name'].lower():
                        wandb_procs.append(proc)
                    elif proc.info['cmdline'] and any('wandb' in str(cmd).lower() for cmd in proc.info['cmdline']):
                        wandb_procs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Update our tracking list
            self.wandb_processes = [p for p in wandb_procs if p.is_running()]
            
            process_count = len(self.wandb_processes)
            if process_count > 0:
                logger.debug(f"Found {process_count} WandB background processes")
            
            return process_count
            
        except Exception as e:
            logger.debug(f"Failed to monitor WandB processes: {e}")
            return 0

    
    
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
        """Finish the current wandb run."""
        # Wait on any pending async checkpoint uploads before finishing
        try:
            if getattr(self, '_pending_checkpoint_futures', None):
                for fut in list(self._pending_checkpoint_futures):
                    try:
                        fut.result(timeout=60)  # best-effort wait
                    except Exception as e:
                        logger.debug(f"Async checkpoint future ended with error (ignored at finish): {e}")
        except Exception:
            pass
        # Shutdown executor
        try:
            if getattr(self, '_io_executor', None):
                self._io_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        if hasattr(self, 'run') and self.run:
            # Auto-cleanup test artifacts before finishing
            try:
                if self.run and hasattr(self.run, 'name'):
                    run_name = self.run.name.lower()
                    if any(keyword in run_name for keyword in ['test', 'async-test', 'cleanup-test']):
                        logger.info("🧹 Auto-cleaning test artifacts before finish")
                        self.cleanup_test_artifacts()
            except Exception as e:
                logger.debug(f"Auto-cleanup failed (non-critical): {e}")
                
            self.run.finish()       

        # Clean up WandB API object to prevent memory accumulation
        if hasattr(self, 'wandb_api') and self.wandb_api:
            del self.wandb_api
            self.wandb_api = None

        # Force garbage collection
        import gc
        gc.collect()

        self.initialized = False
        logger.info("Wandb run finished and resources cleaned up")

    def cleanup_test_artifacts(self, force_cleanup_all: bool = False) -> None:
        """
        Clean up test artifacts to prevent wandb pollution.
        This should be called after tests to remove all created artifacts.
        
        Args:
            force_cleanup_all: If True, removes ALL artifacts from the current run
        """
        if not self.enabled or not hasattr(self, 'run') or not self.run:
            return
            
        try:
            # Get current run name to identify test artifacts
            run_name = self.run.name if self.run else None
            if not run_name:
                return
                
            # Check if this looks like a test run
            is_test_run = any(keyword in run_name.lower() for keyword in ['test', 'async-test', 'cleanup-test'])
            
            if not is_test_run and not force_cleanup_all:
                logger.info(f"Skipping cleanup for non-test run: {run_name}")
                return
                
            logger.info(f"🧹 Cleaning up test artifacts for run: {run_name}")
            
            # Clean up all artifacts created in this run
            try:
                # Get all artifacts from this run - try multiple methods
                artifacts = []
                cleanup_count = 0
                
                # Method 1: Check logged artifacts from current run
                if hasattr(self.run, 'logged_artifacts'):
                    try:
                        artifacts = list(self.run.logged_artifacts())
                        logger.debug(f"Found {len(artifacts)} logged artifacts")
                    except Exception as e:
                        logger.debug(f"Could not get logged artifacts: {e}")
                
                # Method 2: Use wandb API to find artifacts by run ID
                if not artifacts:
                    try:
                        api = self.wandb_api or self._login_and_get_api()
                        if api and hasattr(self.run, 'id'):
                            # Get all artifacts for this specific run
                            run_artifacts = api.run(f"{self.run.entity}/{self.run.project}/{self.run.id}").logged_artifacts()
                            artifacts = list(run_artifacts)
                            logger.debug(f"Found {len(artifacts)} artifacts via API for run {self.run.id}")
                    except Exception as e:
                        logger.debug(f"Could not get artifacts via API: {e}")
                
                # Method 3: Try to find artifacts by project and run name patterns
                if not artifacts:
                    try:
                        # Search for artifacts that match test patterns
                        api = self.wandb_api or self._login_and_get_api()
                        if api:
                            # Search by common test artifact names
                            test_patterns = [
                                f"{run_name}_checkpoint",
                                f"{run_name.replace('-', '_')}_checkpoint", 
                                "test_checkpoint",
                                "cleanup_test_checkpoint"
                            ]
                            
                            for pattern in test_patterns:
                                try:
                                    safe_pattern = self._sanitize_artifact_name(pattern)
                                    # Try to delete this pattern directly
                                    self._cleanup_old_checkpoints(safe_pattern, keep_latest=0)
                                    cleanup_count += 1
                                    logger.debug(f"  Cleaned up by pattern: {safe_pattern}")
                                except Exception as e:
                                    logger.debug(f"  Pattern {pattern} not found or already cleaned: {e}")
                    except Exception as e:
                        logger.debug(f"Pattern-based cleanup failed: {e}")
                
                # Method 4: Clean up known artifacts from the logged artifacts list
                for artifact in artifacts:
                    try:
                        # Delete all versions of this artifact based on its type
                        collection_name = artifact.name
                        artifact_type = getattr(artifact, 'type', 'model_checkpoint')
                        
                        if artifact_type == 'model_checkpoint':
                            # Use checkpoint cleanup for model checkpoints
                            safe_name = self._sanitize_artifact_name(collection_name)
                            self._cleanup_old_checkpoints(safe_name, keep_latest=0)
                        else:
                            # For other artifact types, try direct deletion
                            try:
                                artifact.delete()
                                logger.debug(f"  Directly deleted artifact: {collection_name}")
                            except Exception as delete_error:
                                logger.debug(f"  Failed to directly delete artifact {collection_name}: {delete_error}")
                                
                        cleanup_count += 1
                        logger.debug(f"  Cleaned up artifact: {collection_name} (type: {artifact_type})")
                    except Exception as e:
                        logger.debug(f"  Failed to cleanup artifact {getattr(artifact, 'name', 'unknown')}: {e}")
                        
                if cleanup_count > 0:
                    logger.info(f"✅ Cleaned up {cleanup_count} test artifacts")
                else:
                    logger.info("ℹ️  No artifacts found to clean up (may have been auto-cleaned)")
                    
            except Exception as e:
                logger.warning(f"Failed to enumerate artifacts for cleanup: {e}")
                
            # Also try cleanup by common test checkpoint names
            test_checkpoint_patterns = [
                f"{run_name}_checkpoint",
                "test_checkpoint", 
                "test_checkpoint_sync",
                "multi_checkpoint",
                "async-test-run_checkpoint",
                "cleanup-test-run_checkpoint"
            ]
            
            for pattern in test_checkpoint_patterns:
                try:
                    safe_pattern = self._sanitize_artifact_name(pattern)
                    self._cleanup_old_checkpoints(safe_pattern, keep_latest=0)
                    logger.debug(f"  Cleaned up checkpoint pattern: {safe_pattern}")
                except Exception as e:
                    logger.debug(f"  No artifacts found for pattern {pattern}: {e}")
                    
        except Exception as e:
            logger.warning(f"Test artifact cleanup failed: {e}")

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

    def _get_checkpoint_name(self) -> str:
        """
        Get the checkpoint name based on the current run name.
        
        Returns:
            Checkpoint name in format: {run_name}_checkpoint
        """
        if self.run and self.run.name:
            return f"{self.run.name}_checkpoint"
        else:
            # Fallback to default if run not initialized
            return "checkpoint"

    def get_checkpoint_name(self) -> str:
        """
        Public method to get the current checkpoint name.
        
        Returns:
            Checkpoint name in format: {run_name}_checkpoint
        """
        return self._get_checkpoint_name()

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
            
        Returns:
            Loaded model instance or None if loading failed
        """
        
        # Download artifact (guard against missing artifact / membership errors)
        artifact_path = self._construct_artifact_path(collection_name, alias)
        try:
            model_artifact = self.wandb_api.artifact(artifact_path, type="model")
            model_dir = model_artifact.download()
        except Exception as e:
            # Specific wandb errors (CommError / ValueError) indicate the artifact
            # or membership was not found. Log a clear message and return None so
            # callers can handle absence gracefully.
            logger.warning(f"Could not retrieve wandb artifact '{artifact_path}': {e}")
            return None

        model_file_name = None
        for file in model_artifact.files():
            if file.name.endswith('.pth'):
                model_file_name = file.name
                break

        if not model_file_name:
            logger.error(f"No .pth file found in artifact: {artifact_path}")
            return None

        model_path = os.path.join(model_dir, model_file_name)
        logger.info(f"Model path: {model_path}")
        
        if os.path.exists(model_path):
            # Read artifact metadata to get saved model configuration
            saved_config = {}
            try:
                if hasattr(model_artifact, 'metadata') and model_artifact.metadata:
                    model_config_meta = model_artifact.metadata.get('model_config', {})
                    if 'embedding_dim' in model_config_meta:
                        saved_config['embedding_dim'] = model_config_meta['embedding_dim']
                        logger.info(f"Found saved embedding_dim in metadata: {saved_config['embedding_dim']}")
            except Exception as e:
                logger.debug(f"Could not read artifact metadata: {e}")

            # Merge saved config with provided kwargs (saved config takes precedence for critical params)
            merged_kwargs = dict(kwargs)
            merged_kwargs.update(saved_config)
            
            # Initialize model with merged configuration
            model = model_class(**merged_kwargs)

            # Load raw checkpoint (may be a state_dict or a full checkpoint dict)
            raw = torch.load(model_path, map_location=device)
            if isinstance(raw, dict) and 'model_state_dict' in raw:
                state_dict = raw['model_state_dict']
            else:
                state_dict = raw

            model_state = model.state_dict()

            # If the checkpoint contains keys for a lazy-created head, ensure the model
            # has its head constructed before attempting a strict load. This avoids
            # "unexpected key" errors when the model lazily creates the head on first
            # forward.
            try:
                sd_keys = set(state_dict.keys())
            except Exception:
                sd_keys = set()

            if any(k.startswith('backbone._head') for k in sd_keys) and getattr(model.backbone, '_head', None) is None:
                logger.info("Checkpoint contains backbone._head keys; initializing head via model API if available")
                try:
                    # Prefer a model-provided initialization API
                    if hasattr(model, 'ensure_head_initialized'):
                        model.ensure_head_initialized(device=device)
                    else:
                        # Fallback to conservative dummy forward
                        model.to(device)
                        model.eval()
                        with torch.no_grad():
                            dummy = torch.zeros((1, 3, 224, 224), device=device)
                            model(dummy)
                except Exception as e:
                    logger.warning(f"Could not initialize model head before loading: {e}")

            # Use centralized robust loader to handle lazy heads and mismatched keys
            success, unexpected, missing, err = self._robust_load_state_dict(model, state_dict, device=device)
            if success:
                if unexpected:
                    logger.warning(f"State dict contained unexpected keys (they were ignored): {unexpected[:10]}{('...' if len(unexpected)>10 else '')}")
                if missing:
                    logger.info(f"State dict is missing keys (these were left at defaults): {missing[:10]}{('...' if len(missing)>10 else '')}")
                logger.info(f"✓ Successfully loaded model from wandb registry: {collection_name}:{alias}")
                return model
            else:
                logger.error(f"Failed to load model from registry: {err}")
                return None
        else:
            logger.warning(f"Model file not found in wandb artifact: {model_path}")
            return None
        
    @requires_wandb_initialized
    @safe_wandb_operation()
    def save_model_to_registry(self, model: torch.nn.Module, collection_name: str, 
                            alias: str = "latest", file_name: str = "model.pth", 
                            metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> None:
        """
        Save model to wandb model registry with automatic version management.
        Keeps only the last versions plus any versions tagged.

        Args:
            model: PyTorch model to save
            collection_name: Name for the model collection in wandb registry
            alias: Model version alias (e.g., "latest", "best", "v1")
            file_name: Name of the model file to save
            metadata: Additional metadata to include
            tags: Optional list of tags to add to the artifact
        """
        # Build metadata: merge user-provided metadata with detected model config
        auto_meta: Dict[str, Any] = {}
        try:
            # Common high-level attrs
            auto_meta['embedding_dim'] = getattr(model, 'embedding_dim', None)
            auto_meta['dropout'] = getattr(model, 'dropout', getattr(model, 'dropout_rate', None))

            # Try to detect backbone head linear layer shape(s)
            head_info = {}
            backbone = getattr(model, 'backbone', None)
            if backbone is not None and getattr(backbone, '_head', None) is not None:
                head = backbone._head
                # find any Linear modules inside the head
                for name, mod in head.named_modules():
                    if isinstance(mod, torch.nn.Linear):
                        head_info['linear_'+name] = {'in_features': mod.in_features, 'out_features': mod.out_features}
                if head_info:
                    auto_meta['head'] = head_info
        except Exception as e:
            logger.debug(f"Failed to auto-detect model metadata: {e}")

        merged_meta = {} if metadata is None else dict(metadata)
        # Put auto-detected metadata under key 'model_config' for loader consumption
        merged_meta.setdefault('model_config', {})
        merged_meta['model_config'].update(auto_meta)

        # Use the collection name as provided (do not alter collection naming here)
        artifact = wandb.Artifact(
            name=collection_name,
            type="model",
            metadata=merged_meta
        )

        # Save model state dict to a local file and attach to artifact
        model_path = file_name
        torch.save(model.state_dict(), model_path)
        artifact.add_file(model_path)

        # Log artifact to wandb and attempt to link it under a registry path
        logged_artifact = self.run.log_artifact(artifact, type="model", tags=tags)
        target_path = f"wandb-registry-model/{collection_name}"
        try:
            self.run.link_artifact(logged_artifact, target_path=target_path)
        except Exception as e:
            logger.warning(f"Failed to link artifact to {target_path}: {e}")

        logger.info(f"✓ Model saved to wandb registry: {collection_name}")

        # Clean up old model versions (use collection_name as provided)
        try:
            self._cleanup_old_model_versions(collection_name)
        except Exception as e:
            logger.warning(f"Failed to cleanup old model versions for {collection_name}: {e}")

        # Clean up old checkpoints since final model is now saved
        checkpoint_name = self._get_checkpoint_name()
        safe_checkpoint = self._sanitize_artifact_name(checkpoint_name)
        try:
            self._cleanup_old_checkpoints(safe_checkpoint, keep_latest=0)  # Remove all checkpoints
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoints for {safe_checkpoint}: {e}")

        return None

    def _cleanup_old_model_versions(self, collection_name: str, keep_latest: int = 3) -> None:
        """
        Clean up old model versions, keeping only the latest N versions and any marked "do not delete".
        
        Args:
            collection_name: Name of the model artifact
            keep_latest: Number of latest versions to keep (default: 3)
        """
        
        # Get all versions of this model
        artifact_collection_name = f"{wandb_config.team}/{wandb_config.project}/{collection_name}"

        try:
            # List all versions of the artifact
            versions = list(self.wandb_api.artifact_type("model", project=wandb_config.project).collection(collection_name).artifacts())

            if len(versions) <= keep_latest:
                logger.info(f"Only {len(versions)} versions exist, no cleanup needed")
                return
            
            # Sort versions by creation time (newest first)
            versions.sort(key=lambda x: x.created_at, reverse=True)
            
            # Identify versions to delete
            versions_to_delete = []
            for i, version in enumerate(versions):
                # Keep the latest N versions
                if i < keep_latest:
                    continue
                
                # Check if this version has "do not delete" alias or tag
                do_not_delete = False
                
                # Check aliases
                for alias in version.aliases:
                    if alias in wandb_config.model_tags_to_skip_deletion:
                        do_not_delete = True
                        break
                
                # Check metadata for do not delete flag
                if not do_not_delete and version.metadata:
                    if version.metadata.get("do_not_delete", False) or \
                       "do not delete" in str(version.metadata.get("tags", "")).lower():
                        do_not_delete = True
                
                if not do_not_delete:
                    versions_to_delete.append(version)
            
            # Delete old versions
            for version in versions_to_delete:
                try:
                    version.delete()
                    logger.info(f"Deleted old model version: {collection_name}:v{version.version}")
                except Exception as e:
                    logger.warning(f"Failed to delete version {version.version}: {e}")
            
            if versions_to_delete:
                logger.info(f"Cleaned up {len(versions_to_delete)} old model versions")
            else:
                logger.info("No old versions to clean up (all marked as 'do not delete')")
                
        except Exception as e:
            logger.warning(f"Failed to access artifact collection {artifact_collection_name}: {e}")

    @requires_wandb_initialized
    @safe_wandb_operation()
    def update_run_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update the wandb run config and tags with the contents of a dictionary and a list of tags.
        Args:
            config_dict: Dictionary of config values to update in the wandb run
        """
        self.run.config.update(config_dict, allow_val_change=True)
        logger.debug(f"Updated wandb run config with: {list(config_dict.keys())}")


    @requires_wandb_initialized
    @safe_wandb_operation()
    def save_checkpoint(self, epoch: int, model: Optional[torch.nn.Module] = None, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   model_state_dict: Optional[Dict[str, Any]] = None,
                   optimizer_state_dict: Optional[Dict[str, Any]] = None,
                   loss: Optional[float] = None, 
                   model_name: str = "model", model_config: Optional[Dict[str, Any]] = None,
                   async_upload: bool = True,
                   **kwargs) -> Optional[str]:
        """
        Save model checkpoint to wandb and clean up previous versions.
        
        Args:
            epoch: Current epoch number
            model: The model to save (alternative to model_state_dict)
            optimizer: The optimizer to save (alternative to optimizer_state_dict)
            model_state_dict: Pre-computed model state dictionary (discouraged for memory reasons)
            optimizer_state_dict: Pre-computed optimizer state dictionary (discouraged for memory reasons)
            loss: Current loss value
            model_name: Name of the model for artifact naming
            model_config: Optional model configuration metadata
            async_upload: Whether to upload asynchronously in background
            
        Returns:
            Artifact reference (name:latest) or placeholder when async
            
        Note:
            For optimal memory usage, pass model and optimizer objects directly instead of pre-computed state_dicts.
        """
        import os, psutil, gc

        wandb_process_count_before = self._monitor_wandb_processes()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024

        if loss is None:
            loss = kwargs.get('current_loss', None)

        # Normalize legacy / positional argument usage
        try:
            if model is not None and not hasattr(model, 'state_dict') and isinstance(model, dict):
                model_state_dict = model; model = None
            if optimizer is not None and not hasattr(optimizer, 'state_dict') and isinstance(optimizer, dict):
                optimizer_state_dict = optimizer; optimizer = None
            if model_state_dict is not None and isinstance(model_state_dict, (int, float)):
                loss = float(model_state_dict); model_state_dict = None
            if optimizer_state_dict is not None and isinstance(optimizer_state_dict, str):
                model_name = optimizer_state_dict; optimizer_state_dict = None
        except Exception:
            pass

        if model_state_dict is None and model is not None:
            model_state_dict = model.state_dict()
        if optimizer_state_dict is None and optimizer is not None:
            optimizer_state_dict = optimizer.state_dict()

        if model_state_dict is None:
            raise ValueError("Either model or model_state_dict must be provided")
        if optimizer_state_dict is None:
            raise ValueError("Either optimizer or optimizer_state_dict must be provided")

        with tempfile.NamedTemporaryFile(suffix=f'_epoch_{epoch}.pth', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
        try:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'model_config': model_config or {}
            }
            torch.save(checkpoint_data, checkpoint_path)
        finally:
            try: del model_state_dict
            except Exception: pass
            try: del optimizer_state_dict
            except Exception: pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        memory_after = process.memory_info().rss / 1024 / 1024
        if abs(memory_after - memory_before) > 50:
            logger.info(f"Checkpoint save memory: {memory_before:.1f}MB → {memory_after:.1f}MB (Δ{(memory_after-memory_before):+.1f}MB)")

        artifact_name = self._get_checkpoint_name()
        safe_artifact_name = self._sanitize_artifact_name(artifact_name)

        def _upload_and_cleanup(path: str, safe_name: str):
            # Initialize artifact info with defaults
            artifact_info = {
                'name': safe_name,
                'logged_artifact': None,
                'epoch': epoch
            }
            
            try:
                artifact = wandb.Artifact(
                    name=safe_name,
                    type="model_checkpoint",
                    description=f"Model checkpoint for epoch {epoch}",
                    metadata={
                        'epoch': epoch,
                        'loss': loss,
                        'model_name': model_name,
                        'timestamp': datetime.now().isoformat(),
                        'async': bool(async_upload)
                    }
                )
                artifact.add_file(path, name=f"checkpoint_epoch_{epoch}.pth")
                logged_artifact = self.run.log_artifact(artifact)
                
                # Update artifact info with successful upload
                artifact_info['logged_artifact'] = logged_artifact
                
                # Manage the 'latest' tag for checkpoints
                try:
                    self._manage_latest_checkpoint_tag(safe_name, logged_artifact)
                except Exception as e:
                    logger.debug(f"Latest tag management failed for {safe_name}: {e}")
                
            except Exception as e:
                logger.warning(f"Checkpoint upload failed for {safe_name}: {e}")
            finally:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception: pass
                try: del artifact  # type: ignore
                except Exception: pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Cleanup old checkpoints with retry logic to handle timing issues
                try:
                    self._cleanup_old_checkpoints_with_retry(safe_name, artifact_info['epoch'])
                except Exception as e:
                    logger.debug(f"cleanup_old_checkpoints failed for {safe_name}: {e}")

        if async_upload and getattr(self, '_io_executor', None):
            try:
                fut = self._io_executor.submit(_upload_and_cleanup, checkpoint_path, safe_artifact_name)
                self._pending_checkpoint_futures.append(fut)
                self._pending_checkpoint_futures = [f for f in self._pending_checkpoint_futures if not f.done()]
                logger.info(f"Scheduled async checkpoint upload for epoch {epoch} -> {safe_artifact_name}")
                return f"{safe_artifact_name}:pending"
            except Exception as e:
                logger.debug(f"Async scheduling failed, falling back to sync: {e}")

        _upload_and_cleanup(checkpoint_path, safe_artifact_name)
        wandb_process_count_after = self._monitor_wandb_processes()
        if wandb_process_count_after > wandb_process_count_before + 2:
            logger.info(f"WandB spawned {wandb_process_count_after - wandb_process_count_before} additional processes during checkpoint save - normal for uploads")
        logger.info(f"Saved model checkpoint to wandb for epoch {epoch}")
        return f"{safe_artifact_name}:latest"

    def _cleanup_old_checkpoints(self, artifact_name: str, keep_latest: int = 1):
        """
        Clean up old wandb checkpoint artifacts, keeping only the latest N versions.

        Args:
            artifact_name: Name of the artifact to clean up
            keep_latest: Number of latest versions to keep (default: 1)
        """
        try:
            collection_name = artifact_name
            logger.debug(f"Searching for checkpoint artifacts in project={wandb_config.project}, collection={collection_name}")

            artifact_type_api = self.wandb_api.artifact_type("model_checkpoint", project=wandb_config.project)
            artifact_collection = artifact_type_api.collection(collection_name)

            # Fast path: if caller wants to remove ALL versions, attempt a single
            # collection-level delete (cheap) and return.
            if keep_latest == 0:
                try:
                    artifact_collection.delete()
                    logger.info(f"Deleted entire checkpoint collection {collection_name} (keep_latest=0)")
                    return
                except Exception as e:
                    # Fall back to per-artifact deletion below if collection delete fails
                    logger.debug(f"Collection delete fast-path failed for {collection_name}: {e}; falling back to per-version deletion")

            # Stream artifacts without materializing all metadata up-front.
            # We keep only the `keep_latest` newest artifacts in a min-heap by created_at
            import heapq
            keep_heap: List[tuple] = []  # (created_at, artifact)
            to_delete: List[Any] = []

            for art in artifact_collection.artifacts():  # generator
                created = getattr(art, 'created_at', None)
                if created is None:
                    # If no created_at, treat as oldest; mark for deletion unless within keep capacity
                    if keep_latest > 0 and len(keep_heap) < keep_latest:
                        heapq.heappush(keep_heap, (created, art))
                    else:
                        to_delete.append(art)
                    continue
                if keep_latest == 0:
                    to_delete.append(art)
                    continue
                if len(keep_heap) < keep_latest:
                    heapq.heappush(keep_heap, (created, art))
                else:
                    # Smallest (oldest) is at index 0; if current is newer, replace oldest
                    if created > keep_heap[0][0]:
                        _, oldest = heapq.heapreplace(keep_heap, (created, art))
                        to_delete.append(oldest)
                    else:
                        to_delete.append(art)

            total_found = len(keep_heap) + len(to_delete)
            logger.debug(f"Found {total_found} checkpoint artifacts for {artifact_name}")

            if total_found <= keep_latest:
                logger.debug(f"Only {total_found} versions exist, no cleanup needed for {artifact_name}")
                # Defensive cleanup
                del keep_heap, to_delete
                import gc; gc.collect()
                return

            deleted_count = 0
            for version in to_delete:
                try:
                    logger.debug(f"Preparing to delete artifact version: {getattr(version, 'name', '<unknown>')} (created: {getattr(version, 'created_at', '<unknown>')})")
                    # Alias removal (only if aliases present; skip heavy logic otherwise)
                    try:
                        aliases = getattr(version, 'aliases', None)
                        if aliases:
                            remove_fn = getattr(version, 'remove', None)
                            if callable(remove_fn):
                                for a in list(aliases):
                                    try:
                                        remove_fn(a)
                                    except Exception:
                                        pass
                            else:
                                try:
                                    version.aliases = []  # type: ignore
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    version.delete()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoint {getattr(version, 'name', '<unknown>')}: {e}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old checkpoint artifacts for {artifact_name}")

            # Fallback: if nothing deleted and keep_latest==0 (collection delete failed earlier)
            if deleted_count == 0 and keep_latest == 0:
                try:
                    artifact_collection.delete()
                    logger.info(f"Deleted artifact collection {collection_name} as fallback cleanup (second attempt)")
                except Exception as e:
                    logger.warning(f"Fallback collection delete failed for {collection_name}: {e}")

            # Explicit cleanup of local references
            del keep_heap, to_delete
            import gc; gc.collect()

        except Exception as e:
            # Bubble up errors so callers can see when cleanup genuinely failed.
            logger.error(f"Failed to cleanup checkpoints for {artifact_name}: {e}")
            raise

    def _cleanup_old_checkpoints_with_retry(self, artifact_name: str, current_epoch: int, keep_latest: int = 1, max_retries: int = 5):
        """
        Cleanup old checkpoints with retry logic to handle timing issues.
        
        This method waits for the artifact upload to complete and become visible 
        in wandb before attempting cleanup. This ensures that the cleanup sees 
        the newly uploaded artifact and doesn't delete the wrong ones.
        
        Args:
            artifact_name: Name of the checkpoint artifact collection
            current_epoch: The epoch number that was just uploaded
            keep_latest: Number of latest versions to keep (default: 1)
            max_retries: Maximum number of retry attempts (default: 5)
        """
        import time
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Get the artifact collection to check current state
                collection_name = f"{self.run.entity}/{self.run.project}/{artifact_name}"
                try:
                    artifact_collection = self.wandb_api.artifact_collection("model_checkpoint", collection_name)
                    artifacts = list(artifact_collection.artifacts())
                    
                    # Check if we can find the newly uploaded artifact
                    current_epoch_found = False
                    for artifact in artifacts:
                        try:
                            # Check metadata for current epoch
                            metadata = getattr(artifact, 'metadata', {}) or {}
                            if metadata.get('epoch') == current_epoch:
                                current_epoch_found = True
                                break
                        except Exception:
                            # If metadata check fails, try name-based detection
                            if f"epoch_{current_epoch}" in getattr(artifact, 'name', ''):
                                current_epoch_found = True
                                break
                    
                    if current_epoch_found or retry_count >= max_retries - 1:
                        # Either found the new artifact or max retries reached, proceed with cleanup
                        logger.debug(f"Artifact for epoch {current_epoch} {'found' if current_epoch_found else 'not found but proceeding'}, running cleanup (attempt {retry_count + 1})")
                        self._cleanup_old_checkpoints(artifact_name, keep_latest)
                        return
                    else:
                        # New artifact not visible yet, wait and retry
                        logger.debug(f"New artifact for epoch {current_epoch} not yet visible, retrying in {0.5 * (retry_count + 1)}s (attempt {retry_count + 1}/{max_retries})")
                        time.sleep(0.5 * (retry_count + 1))  # Exponential backoff
                        retry_count += 1
                        
                except Exception as e:
                    # If we can't access the collection, fall back to regular cleanup
                    logger.debug(f"Could not verify artifact upload completion: {e}, proceeding with cleanup")
                    self._cleanup_old_checkpoints(artifact_name, keep_latest)
                    return
                    
            except Exception as e:
                logger.debug(f"Retry cleanup attempt {retry_count + 1} failed: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    # Final fallback to regular cleanup
                    logger.debug(f"Max retries reached, falling back to regular cleanup")
                    try:
                        self._cleanup_old_checkpoints(artifact_name, keep_latest)
                    except Exception as fallback_error:
                        logger.debug(f"Fallback cleanup also failed: {fallback_error}")
                    return
                time.sleep(0.5 * retry_count)

    def _manage_latest_checkpoint_tag(self, new_artifact_name: str, new_artifact: Any) -> None:
        """
        Manage the 'latest' tag for checkpoints.
        
        Adds 'latest' tag to the new checkpoint and removes it from any previous checkpoints.
        
        Args:
            new_artifact_name: Name of the newly created artifact
            new_artifact: The logged artifact object
        """
        try:
            # First, add 'latest' tag to the new artifact
            if new_artifact:
                try:
                    new_artifact.aliases.append('latest')
                    logger.debug(f"Added 'latest' tag to checkpoint: {new_artifact_name}")
                except Exception as e:
                    logger.debug(f"Could not add 'latest' tag to new artifact: {e}")
            
            # Then, find and remove 'latest' tag from other checkpoints
            try:
                # Get the artifact collection
                collection_name = f"{self.run.entity}/{self.run.project}/{new_artifact_name}"
                artifact_collection = self.wandb_api.artifact_collection("model_checkpoint", collection_name)
                
                # Iterate through all artifacts in the collection
                for artifact in artifact_collection.artifacts():
                    # Skip the new artifact we just created
                    if artifact.name == new_artifact_name:
                        continue
                        
                    # Check if this artifact has the 'latest' tag
                    try:
                        aliases = getattr(artifact, 'aliases', [])
                        if 'latest' in aliases:
                            # Remove the 'latest' tag
                            try:
                                aliases.remove('latest')
                                artifact.aliases = aliases
                                logger.debug(f"Removed 'latest' tag from previous checkpoint: {artifact.name}")
                            except Exception as e:
                                logger.debug(f"Could not remove 'latest' tag from {artifact.name}: {e}")
                    except Exception as e:
                        logger.debug(f"Could not check aliases for {artifact.name}: {e}")
                        
            except Exception as e:
                logger.debug(f"Could not access artifact collection for tag management: {e}")
                
        except Exception as e:
            logger.debug(f"Latest tag management failed: {e}")

    @requires_wandb_enabled
    @safe_wandb_operation()
    def load_checkpoint(self, artifact_name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Load model checkpoint from wandb.
        
        Args:
            artifact_name: Name of the wandb artifact
            version: Version of the artifact to load (default: "latest")
            
        Returns:
            Checkpoint data dictionary or None if failed
        """
        
        # Download artifact
        artifact_path = self._construct_artifact_path(artifact_name, version)
        artifact_dir = None
        try:
            if self.wandb_api.artifact_exists(artifact_path):
                artifact = self.wandb_api.artifact(artifact_path, type="model_checkpoint")
                artifact_dir = artifact.download()
        except Exception as e:
            logger.info(f"Artifact {artifact_name}:{version} not found")
            return None

        if artifact_dir is None:
            logger.error(f"Failed to download artifact {artifact_name}:{version}")
            return None

        # Find checkpoint file
        checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
        
        if not checkpoint_files:
            logger.error(f"No checkpoint files found in artifact {artifact_name}:{version}")
            return None
        
        # Load the checkpoint (assume first .pth file)
        checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        logger.info(f"Loaded checkpoint from wandb: {artifact_name}:{version} (epoch {checkpoint_data.get('epoch', 'unknown')})")
        return checkpoint_data


    def resume_training_from_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, artifact_name: str, version: str = "latest") -> int:
        """
        Resume training from a wandb checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into  
            artifact_name: Name of the wandb checkpoint artifact
            version: Version to load (default: "latest")
            
        Returns:
            Starting epoch number for resuming training
        """
        checkpoint_data = self.load_checkpoint(artifact_name, version)
        
        if checkpoint_data is None:
            logger.warning("Could not load checkpoint, starting training from epoch 1")
            return 1
        
        try:
            # Prepare state_dict (checkpoint may be a dict containing model_state_dict)
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
            else:
                state_dict = checkpoint_data

            # Defensive: ensure lazy-created backbone head exists if checkpoint contains its keys
            try:
                sd_keys = set(state_dict.keys())
            except Exception:
                sd_keys = set()

            try:
                model_backbone = getattr(model, 'backbone', None)
            except Exception:
                model_backbone = None

            # If checkpoint has backbone._head.* keys but model's backbone hasn't created _head yet,
            # attempt to initialize it via model API or a dummy forward so load_state_dict succeeds.
            if any(k.startswith('backbone._head') for k in sd_keys) and model_backbone is not None and getattr(model_backbone, '_head', None) is None:
                logger.info("Checkpoint appears to contain backbone._head keys; attempting to initialize model head before loading checkpoint")
                try:
                    if hasattr(model, 'ensure_head_initialized'):
                        # Prefer model-provided convenience method
                        try:
                            device_str = str(next(model.parameters()).device)
                        except Exception:
                            device_str = 'cpu'
                        model.ensure_head_initialized(device=device_str)
                    else:
                        # Fallback: run a conservative dummy forward on CPU/device of model
                        try:
                            dev = next(model.parameters()).device
                        except Exception:
                            dev = torch.device('cpu')
                        model.to(dev)
                        model.eval()
                        with torch.no_grad():
                            dummy = torch.zeros((1, 3, 224, 224), device=dev)
                            model(dummy)
                except Exception as e:
                    logger.warning(f"Could not initialize model head before loading checkpoint: {e}")

            model_state = model.state_dict()

            # Use centralized robust loader to handle lazy heads and mismatched keys
            success, unexpected, missing, err = self._robust_load_state_dict(model, state_dict)
            if not success:
                logger.error(f"Failed to load model state from checkpoint: {err}")
                return 1
            if unexpected:
                logger.warning(f"State dict contained unexpected keys (they were ignored): {unexpected[:10]}{('...' if len(unexpected)>10 else '')}")
            if missing:
                logger.info(f"State dict is missing keys (these were left at defaults): {missing[:10]}{('...' if len(missing)>10 else '')}")

            # Load optimizer state if present
            try:
                if isinstance(checkpoint_data, dict) and 'optimizer_state_dict' in checkpoint_data:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state from checkpoint: {e}")

            # Get starting epoch
            start_epoch = checkpoint_data.get('epoch', 0) + 1 if isinstance(checkpoint_data, dict) else 1
            logger.info(f"Resumed training from epoch {start_epoch}")
            return start_epoch

        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return 1


    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_training_metrics(self, epoch: int, train_loss: float, val_loss: Optional[float] = None, 
                        learning_rate: Optional[float] = None, **kwargs) -> None:
        """
        Log training metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for the epoch
            val_loss: Optional validation loss
            learning_rate: Current learning rate
            **kwargs: Additional metrics to log
        """
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

    @requires_wandb_initialized
    @safe_wandb_operation()
    def log_batch_metrics(self, batch_idx: int, epoch: int, batch_loss: float, **kwargs) -> None:
        """
        Log batch-level metrics during training.
        
        Args:
            batch_idx: Current batch index
            epoch: Current epoch number
            batch_loss: Loss for this batch
            **kwargs: Additional batch metrics to log
        """
        metrics = {
            "batch_loss": batch_loss,
            "epoch": epoch,
            "batch": batch_idx,
            **kwargs
        }
        
        self.run.log(metrics)

    def _sanitize_artifact_name(self, name: Optional[str]) -> str:
        """
        Sanitize artifact/model/checkpoint names so they only contain
        allowed characters for wandb artifact names: alphanumeric, dash,
        underscore and dot.

        Any disallowed character is replaced with an underscore. If the
        resulting name is empty, return a safe default.
        """
        if not name:
            return "artifact"
        # Replace any sequence of invalid chars with a single underscore
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))
        # Trim leading/trailing underscores or dots
        safe = safe.strip('_.-')
        if not safe:
            return "artifact"
        if safe != name:
            logger.info(f"Sanitized artifact name '{name}' -> '{safe}'")
        return safe


wandb_logger = WandbLogger()
            