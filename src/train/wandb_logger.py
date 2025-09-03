import os
import logging
import tempfile
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional, List, Callable, Tuple
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

    def _cleanup_wandb_processes(self, force: bool = False) -> int:
        """
        Monitor WandB processes for issues but DO NOT terminate them.
        WandB processes should run for the entire training session.

        Args:
            force: If True, would terminate processes (but we never do this for WandB)

        Returns:
            Number of problematic processes found (but not terminated)
        """
        try:
            process_count = self._monitor_wandb_processes()
            if process_count == 0:
                return 0

            problematic_count = 0

            for proc in self.wandb_processes[:]:  # Copy list to avoid modification during iteration
                try:
                    # Only check for truly problematic processes (zombies)
                    # We DO NOT terminate WandB processes as they need to run for the entire session
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        logger.warning(f"Found zombie WandB process {proc.pid} ({proc.name()}) - this should be cleaned up by WandB")
                        problematic_count += 1
                    elif self._is_process_old(proc):
                        logger.debug(f"WandB process {proc.pid} has been running for a while - this is normal")

                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"Could not check WandB process {proc.pid}: {e}")
                    # Remove from our list if process no longer exists
                    try:
                        self.wandb_processes.remove(proc)
                    except ValueError:
                        pass
                except Exception as e:
                    logger.debug(f"Error checking WandB process {proc.pid}: {e}")

            if problematic_count > 0:
                logger.warning(f"Found {problematic_count} problematic WandB processes (zombies) - these should be handled by WandB's cleanup")

            return problematic_count

        except Exception as e:
            logger.debug(f"Failed to monitor WandB processes: {e}")
            return 0

    def periodic_cleanup(self) -> int:
        """
        Perform periodic cleanup of WandB resources and processes.
        This should be called periodically during training to prevent accumulation.
        
        Returns:
            Number of processes cleaned up
        """
        if not self.enabled:
            return 0
            
        try:
            # Clean up WandB background processes
            cleaned = self._cleanup_wandb_processes(force=False)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return cleaned
            
        except Exception as e:
            logger.debug(f"Failed to perform periodic cleanup: {e}")
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
            "reinit": "return_previous"
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
        if hasattr(self, 'run') and self.run:
            self.run.finish()

        # Monitor WandB processes before finishing (don't terminate them)
        logger.info("Monitoring WandB background processes before finish...")
        problematic_processes = self._cleanup_wandb_processes(force=False)  # Just monitor, don't terminate
        if problematic_processes > 0:
            logger.warning(f"Found {problematic_processes} problematic WandB processes - WandB should handle cleanup")
        else:
            logger.info("WandB processes appear healthy")

        # Clean up WandB API object to prevent memory accumulation
        if hasattr(self, 'wandb_api') and self.wandb_api:
            del self.wandb_api
            self.wandb_api = None

        # Force garbage collection
        import gc
        gc.collect()

        self.initialized = False
        logger.info("Wandb run finished and resources cleaned up")

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
                   model_name: str = "model", model_config: Optional[Dict[str, Any]] = None, **kwargs) -> Optional[str]:
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
            
        Returns:
            Path to saved checkpoint artifact, or None if failed
            
        Note:
            For optimal memory usage, pass model and optimizer objects directly instead of pre-computed state_dicts.
        """
        import psutil
        import os

        # Monitor WandB processes before checkpoint saving
        wandb_process_count_before = self._monitor_wandb_processes()

        # Monitor memory usage before and after checkpoint operations
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Accept either `loss` or legacy `current_loss` kwarg
        if loss is None:
            loss = kwargs.get('current_loss', None)

        # Normalize legacy / positional argument usage.
        # Some older tests/callers pass state_dicts positionally (model, optimizer)
        # or pass loss/model_name in place of later params. Detect common
        # misbindings and normalize into the explicit variables below.
        try:
            # If `model` is actually a state_dict (dict) instead of an nn.Module
            if model is not None and not hasattr(model, 'state_dict') and isinstance(model, dict):
                model_state_dict = model
                model = None

            # If `optimizer` is actually an optimizer state dict
            if optimizer is not None and not hasattr(optimizer, 'state_dict') and isinstance(optimizer, dict):
                optimizer_state_dict = optimizer
                optimizer = None

            # If model_state_dict was accidentally given as a numeric loss (legacy positional),
            # shift it into `loss` and clear the dict variable.
            if model_state_dict is not None and isinstance(model_state_dict, (int, float)):
                loss = float(model_state_dict)
                model_state_dict = None

            # If optimizer_state_dict is a string it's likely the legacy `model_name` positional arg
            if optimizer_state_dict is not None and isinstance(optimizer_state_dict, str):
                model_name = optimizer_state_dict
                optimizer_state_dict = None
        except Exception:
            # Be defensive: if normalization fails, continue and let later validations raise
            pass

        # Create state dictionaries efficiently with immediate scope management
        if model_state_dict is None and model is not None:
            # Create model state dict in limited scope - create a copy to avoid holding model references
            model_state_dict = {k: v.clone().detach() for k, v in model.state_dict().items()}

        if optimizer_state_dict is None and optimizer is not None:
            # Create optimizer state dict in limited scope - deep copy to avoid optimizer references
            import copy
            optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            
        # Validate we have the required state
        if model_state_dict is None:
            raise ValueError("Either model or model_state_dict must be provided")
        if optimizer_state_dict is None:
            raise ValueError("Either optimizer or optimizer_state_dict must be provided")

        # Create temporary checkpoint file - MEMORY EFFICIENT VERSION
        with tempfile.NamedTemporaryFile(suffix=f'_epoch_{epoch}.pth', delete=False) as tmp_file:
            checkpoint_path = tmp_file.name
            
        try:
            # Save checkpoint data directly to file without creating intermediate dict
            # This avoids holding 2x the tensor memory (original + dict copy)
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'model_config': model_config or {}
            }
            torch.save(checkpoint_data, checkpoint_path)
            
            # Immediately clear the checkpoint_data dict to free memory
            del checkpoint_data
            
        finally:
            # Clear state dicts from memory immediately after torch.save
            # Clear any state dict locals to avoid holding large tensors in scope
            if 'model_state_dict' in locals():
                try:
                    del model_state_dict
                except Exception:
                    pass
            if 'optimizer_state_dict' in locals():
                try:
                    del optimizer_state_dict
                except Exception:
                    pass
            
            # Force garbage collection to ensure tensor memory is released
            import gc
            gc.collect()
            # Additional cleanup for PyTorch CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Monitor memory after checkpoint save
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        if abs(memory_delta) > 50:  # Log if memory changed by more than 50MB
            logger.info(f"Checkpoint save memory: {memory_before:.1f}MB → {memory_after:.1f}MB (Δ{memory_delta:+.1f}MB)")
        
        # Create wandb artifact
        artifact_name = self._get_checkpoint_name()
        safe_artifact_name = self._sanitize_artifact_name(artifact_name)
        artifact = wandb.Artifact(
            name=safe_artifact_name,
            type="model_checkpoint",
            description=f"Model checkpoint for epoch {epoch}",
            metadata={
                "epoch": epoch,
                "loss": loss,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Add checkpoint file to artifact
            artifact.add_file(checkpoint_path, name=f"checkpoint_epoch_{epoch}.pth")

            # Log artifact to wandb (this automatically versions it)
            self.run.log_artifact(artifact)
            
        except Exception:
            # Ensure we always clean up local resources even on failure
            logger.warning(f"Failed to execute log_artifact for checkpoint {safe_artifact_name}")
            # Re-raise so the safe_wandb_operation wrapper can decide how to propagate
            raise
        finally:
            # Explicitly clean up artifact object to prevent memory accumulation
            try:
                del artifact
            except Exception:
                pass

            # Clean up temporary file if it still exists
            try:
                if os.path.exists(checkpoint_path):
                    os.unlink(checkpoint_path)
            except Exception as e:
                logger.debug(f"Failed to remove temp checkpoint file {checkpoint_path}: {e}")

            # Force garbage collection after artifact operations
            import gc
            gc.collect()
            # Clear CUDA cache if available to free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Monitor WandB processes after checkpoint operations (don't terminate them)
        wandb_process_count_after = self._monitor_wandb_processes()
        if wandb_process_count_after > wandb_process_count_before + 2:  # If we gained 2+ processes
            logger.info(f"WandB spawned {wandb_process_count_after - wandb_process_count_before} additional processes during checkpoint save - this is normal for artifact uploads")

        logger.info(f"Saved model checkpoint to wandb for epoch {epoch}")
        # Clean up previous checkpoint artifacts (keep only latest) using sanitized name
        try:
            self._cleanup_old_checkpoints(safe_artifact_name)
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoints for {safe_artifact_name}: {e}")

        return f"{safe_artifact_name}:latest"


    def _cleanup_old_checkpoints(self, artifact_name: str, keep_latest: int = 1):
        """
        Clean up old wandb checkpoint artifacts, keeping only the latest N versions.

        Args:
            artifact_name: Name of the artifact to clean up
            keep_latest: Number of latest versions to keep (default: 1)
        """
        try:
            # Use the same artifact_type -> collection -> artifacts() pattern as
            # used in `_cleanup_old_model_versions` to list checkpoint artifacts.
            collection_name = artifact_name
            logger.debug(f"Searching for checkpoint artifacts in project={wandb_config.project}, collection={collection_name}")

            artifact_type_api = self.wandb_api.artifact_type("model_checkpoint", project=wandb_config.project)
            artifact_collection = artifact_type_api.collection(collection_name)
            artifact_versions = list(artifact_collection.artifacts())

            logger.debug(f"Found {len(artifact_versions)} checkpoint artifacts for {artifact_name}")

            # Sort by creation time (newest first)
            artifact_versions.sort(key=lambda x: x.created_at, reverse=True)

            if len(artifact_versions) <= keep_latest:
                logger.debug(f"Only {len(artifact_versions)} versions exist, no cleanup needed for {artifact_name}")
                return

            versions_to_delete = artifact_versions[keep_latest:]

            deleted_count = 0
            for version in versions_to_delete:
                try:
                    logger.debug(f"Preparing to delete artifact version: {getattr(version, 'name', '<unknown>')} (created: {getattr(version, 'created_at', '<unknown>')})")

                    # If the version has aliases (for example `latest`), remove them first
                    try:
                        aliases = getattr(version, 'aliases', None)
                        if aliases:
                            logger.debug(f"Removing aliases {aliases} from version {getattr(version, 'name', '<unknown>')}")
                            remove_fn = getattr(version, 'remove', None)
                            # Some wandb SDK versions expose `remove(alias)`; try to call it for each alias.
                            if callable(remove_fn):
                                for a in list(aliases):
                                    try:
                                        remove_fn(a)
                                        logger.debug(f"Removed alias '{a}' from version {getattr(version, 'name', '<unknown>')}")
                                    except Exception as e:
                                        logger.debug(f"Failed to remove alias '{a}' via version.remove(): {e}")
                            else:
                                # Fallback: try to clear the aliases attribute if writable
                                try:
                                    version.aliases = []
                                    logger.debug(f"Cleared aliases on version object for {getattr(version, 'name', '<unknown>')}")
                                except Exception:
                                    logger.debug("No supported alias-removal API available on this wandb SDK; proceeding to delete and letting server handle conflicts")
                    except Exception as e:
                        logger.debug(f"Error while attempting to remove aliases: {e}")

                    # Now attempt deletion
                    logger.debug(f"Deleting artifact version: {getattr(version, 'name', '<unknown>')}")
                    version.delete()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoint {getattr(version, 'name', '<unknown>')}: {e}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old checkpoint artifacts for {artifact_name}")

            # If we couldn't delete any versions (for example due to server-side
            # alias protection), and caller requested removing all versions
            # (keep_latest == 0), attempt to remove the entire collection as a
            # last-resort fallback. This is necessary because some wandb server
            # configurations prevent deleting a version that still has an alias
            # (e.g. 'latest'). Deleting the collection will remove all versions.
            if deleted_count == 0 and keep_latest == 0:
                try:
                    logger.debug(f"Attempting to delete artifact collection {collection_name} as fallback (keep_latest=0)")
                    # artifact_collection variable references the collection API object
                    artifact_collection.delete()
                    logger.info(f"Deleted artifact collection {collection_name} as fallback cleanup")
                except Exception as e:
                    logger.warning(f"Fallback collection delete failed for {collection_name}: {e}")

            # Explicitly clean up artifact versions list to prevent memory accumulation
            del artifact_versions, versions_to_delete
            import gc
            gc.collect()

        except Exception as e:
            # Bubble up errors so callers can see when cleanup genuinely failed.
            logger.error(f"Failed to cleanup checkpoints for {artifact_name}: {e}")
            raise


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
        try:
            if self.wandb_api.artifact_exists(artifact_path):
                artifact = self.wandb_api.artifact(artifact_path, type="model_checkpoint")
                artifact_dir = artifact.download()
        except Exception as e:
            logger.info(f"Artifact {artifact_name}:{version} not found")
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
            