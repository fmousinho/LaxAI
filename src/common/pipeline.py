"""
Generic pipeline base class for multi-step processing workflows.

This module provides a base Pipeline class that can be extended for specific
pipeline implementations, providing common functionality for step execution,
logging, error handling, and intermediate result saving.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .pipeline_step import PipelineStep, StepStatus
from .google_storage import GoogleStorageClient

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Enum for pipeline status values."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class Pipeline:
    """
    Generic base class for multi-step processing pipelines.
    
    This class provides common functionality for:
    - Step execution with error handling and timing
    - Verbose logging
    - Intermediate result saving
    - Pipeline status tracking
    - Step summary reporting
    """
    
    def __init__(self, 
                 pipeline_name: str,
                 storage_client: GoogleStorageClient,
                 step_definitions: Dict[str, Dict[str, Any]],
                 verbose: bool = False, 
                 save_intermediate: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            pipeline_name: Name of the pipeline for identification
            storage_client: Google Storage client for saving intermediate results
            step_definitions: Dictionary of step definitions with format:
                {
                    "step_name": {
                        "description": "Step description",
                        "function": callable_function,
                        "dependencies": ["previous_step"]  # optional
                    }
                }
            verbose: Enable verbose logging (default: False)
            save_intermediate: Save intermediate results for each step (default: False)
        """
        self.pipeline_name = pipeline_name
        self.storage_client = storage_client
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.step_definitions = step_definitions
        
        self.run_guid = str(uuid.uuid4())
        self.run_folder = f"process/{pipeline_name}/run_{self.run_guid}"
        self.status = PipelineStatus.NOT_STARTED
        
        # Initialize steps based on definitions
        self.steps = self._initialize_steps()
        
        if self.verbose:
            logger.info(f"Initialized {pipeline_name} pipeline")
            logger.info(f"Run GUID: {self.run_guid}")
            logger.info(f"Verbose mode: {self.verbose}")
            logger.info(f"Save intermediate: {self.save_intermediate}")
            logger.info(f"Steps: {list(self.steps.keys())}")
    
    def _initialize_steps(self) -> Dict[str, PipelineStep]:
        """
        Initialize pipeline steps from step definitions.
        
        Returns:
            Dictionary of step names to PipelineStep objects
        """
        steps = {}
        for step_name, step_config in self.step_definitions.items():
            description = step_config.get("description", f"Execute {step_name}")
            steps[step_name] = PipelineStep(step_name, description)
        return steps
    
    def _log_step(self, step_name: str, message: str, level: str = "info"):
        """Log step message if verbose mode is enabled."""
        if self.verbose:
            getattr(logger, level)(f"[{self.pipeline_name}:{step_name}] {message}")
    
    def _save_step_output(self, step_name: str, data: Any, context_id: Optional[str] = None, filename: Optional[str] = None):
        """
        Save step output if save_intermediate is enabled.
        
        Args:
            step_name: Name of the step
            data: Data to save
            context_id: Optional context ID (e.g., video_guid, batch_id)
            filename: Optional custom filename
        """
        if not self.save_intermediate:
            return None
        
        try:
            # Determine the output path
            if context_id:
                base_path = f"{self.run_folder}/{context_id}/intermediate"
            else:
                base_path = f"{self.run_folder}/intermediate"
            
            if filename:
                output_path = f"{base_path}/{step_name}_{filename}"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{base_path}/{step_name}_{timestamp}.json"
            
            # Serialize the data
            if isinstance(data, (dict, list)):
                # For dictionaries, filter out large data keys before serialization
                if isinstance(data, dict):
                    filtered_data = {}
                    for key, value in data.items():
                        # Skip keys that contain large data or non-serializable objects
                        if key in ["frames_data", "detections", "loaded_video", "crops_in_memory", 
                                  "modified_crops_in_memory", "augmented_crops_in_memory"]:
                            filtered_data[key] = f"<{key}_excluded_for_serialization>"
                        else:
                            filtered_data[key] = self._make_json_serializable(value)
                    content = json.dumps(filtered_data, indent=2, default=str)
                else:
                    # For lists, use the serialization method directly
                    content = json.dumps(self._make_json_serializable(data), indent=2, default=str)
            else:
                content = str(data)
            
            # Save to storage
            if self.storage_client.upload_from_string(output_path, content):
                self._log_step(step_name, f"Saved output to {output_path}")
                return output_path
            else:
                self._log_step(step_name, f"Failed to save output to {output_path}", "warning")
                return None
        except Exception as e:
            self._log_step(step_name, f"Error saving output: {e}", "error")
            return None
    
    def _execute_step(self, step_name: str, func: Callable, *args, **kwargs):
        """
        Execute a pipeline step with error handling and logging.
        
        Args:
            step_name: Name of the step to execute
            func: Function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function execution
            
        Raises:
            Exception: If the step fails
        """
        if step_name not in self.steps:
            raise ValueError(f"Step '{step_name}' not found in pipeline steps")
        
        step = self.steps[step_name]
        step.start()
        
        self._log_step(step_name, f"Starting: {step.description}")
        
        try:
            result = func(*args, **kwargs)
            
            # Check if the result indicates an error
            if isinstance(result, dict) and result.get("status") == StepStatus.ERROR.value:
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"Step returned error status: {error_msg}")
            
            step.complete(metadata={"args_count": len(args), "kwargs_count": len(kwargs)})
            self._log_step(step_name, f"Completed successfully in {step.duration:.2f}s")
            return result
        except Exception as e:
            step.error(str(e))
            self._log_step(step_name, f"Failed after {step.duration:.2f}s: {e}", "error")
            raise
    
    def _skip_step(self, step_name: str, reason: str):
        """
        Skip a pipeline step with a reason.
        
        Args:
            step_name: Name of the step to skip
            reason: Reason for skipping the step
        """
        if step_name not in self.steps:
            raise ValueError(f"Step '{step_name}' not found in pipeline steps")
        
        step = self.steps[step_name]
        step.skip(reason)
        self._log_step(step_name, f"Skipped: {reason}", "warning")
    
    def _mark_step_completed(self, step_name: str, output_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Manually mark a step as completed (for steps that don't use _execute_step).
        
        Args:
            step_name: Name of the step to mark as completed
            output_path: Optional path to step output
            metadata: Optional metadata about the step
        """
        if step_name not in self.steps:
            raise ValueError(f"Step '{step_name}' not found in pipeline steps")
        
        step = self.steps[step_name]
        step.start()
        step.complete(output_path=output_path, metadata=metadata or {})
        self._log_step(step_name, f"Marked as completed")
    
    def _get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of all pipeline steps."""
        return {
            "pipeline_name": self.pipeline_name,
            "run_guid": self.run_guid,
            "run_folder": self.run_folder,
            "status": self.status.value,
            "steps": {name: step.to_dict() for name, step in self.steps.items()},
            "total_steps": len(self.steps),
            "completed_steps": len([s for s in self.steps.values() if s.status == StepStatus.COMPLETED]),
            "failed_steps": len([s for s in self.steps.values() if s.status == StepStatus.ERROR]),
            "skipped_steps": len([s for s in self.steps.values() if s.status == StepStatus.SKIPPED])
        }
    
    def _create_run_folder(self) -> None:
        """Create the run folder in storage with pipeline metadata."""
        # Create a placeholder file to ensure the folder exists
        placeholder_content = json.dumps({
            "pipeline_name": self.pipeline_name,
            "run_guid": self.run_guid,
            "created_at": datetime.now().isoformat(),
            "verbose": self.verbose,
            "save_intermediate": self.save_intermediate,
            "steps": list(self.steps.keys())
        })
        
        placeholder_blob = f"{self.run_folder}/.pipeline_info.json"
        
        if not self.storage_client.upload_from_string(placeholder_blob, placeholder_content):
            raise RuntimeError(f"Failed to create run folder: {self.run_folder}")
        
        logger.info(f"Created run folder: {self.run_folder}")
    
    def _reset_steps(self):
        """Reset all steps to initial state."""
        for step in self.steps.values():
            step.status = StepStatus.NOT_STARTED
            step.start_time = None
            step.end_time = None
            step.error_message = None
            step.output_path = None
            step.metadata = {}
    
    def _save_checkpoint(self, context: Dict[str, Any], completed_steps: List[str]) -> bool:
        """
        Save pipeline checkpoint to storage.
        
        Args:
            context: Current pipeline context
            completed_steps: List of completed step names
            
        Returns:
            True if checkpoint saved successfully, False otherwise
        """
        try:
            # Exclude non-serializable objects like models from context before saving
            context_to_save = dict(context)
            for key in ["trained_model", "model", "evaluator", "optimizer", "loss_fn"]:
                if key in context_to_save:
                    context_to_save[key] = f"<{key}_excluded_for_serialization>"
            serializable_context = self._make_json_serializable(context_to_save)
                    
            checkpoint_data = {
                "pipeline_name": self.pipeline_name,
                "run_guid": self.run_guid,
                "run_folder": self.run_folder,
                "timestamp": datetime.now().isoformat(),
                "completed_steps": completed_steps,
                "context": serializable_context,
                "steps_summary": {name: step.to_dict() for name, step in self.steps.items()},
                "checkpoint_version": "1.0"
            }
            
            checkpoint_blob = f"{self.run_folder}/.checkpoint.json"
            checkpoint_content = json.dumps(checkpoint_data, indent=2)
            
            success = self.storage_client.upload_from_string(checkpoint_blob, checkpoint_content)
            
            if success:
                logger.info(f"Checkpoint saved: {checkpoint_blob}")
            else:
                logger.error(f"Failed to save checkpoint: {checkpoint_blob}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def _make_json_serializable(self, obj):
        """
        Recursively convert any object to a JSON-serializable format.
        
        Args:
            obj: The object to convert
            
        Returns:
            JSON-serializable representation of the object
        """
        import numpy as np
        
        # Handle None
        if obj is None:
            return None
            
        # Handle basic types
        if isinstance(obj, (str, int, float, bool)):
            return obj
            
        # Handle numpy arrays
        if hasattr(obj, 'tolist'):
            try:
                return obj.tolist()
            except:
                return f"<numpy_array_shape_{getattr(obj, 'shape', 'unknown')}>"
        
        # Handle supervision.Detections objects
        if hasattr(obj, 'xyxy') and hasattr(obj, 'confidence'):
            return f"<supervision_detections_object_len_{len(obj) if hasattr(obj, '__len__') else 'unknown'}>"
        
        # Handle dictionaries
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Skip known problematic keys that contain large data or non-serializable objects
                if key in ["frames_data", "detections", "loaded_video", "crops_in_memory", 
                          "modified_crops_in_memory", "augmented_crops_in_memory"]:
                    result[key] = f"<{key}_excluded_for_serialization>"
                else:
                    result[key] = self._make_json_serializable(value)
            return result
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            try:
                result = []
                for item in obj:
                    result.append(self._make_json_serializable(item))
                return result
            except:
                return f"<list_or_tuple_len_{len(obj) if hasattr(obj, '__len__') else 'unknown'}_excluded>"
        
        # Handle sets
        if isinstance(obj, set):
            try:
                return list(obj)
            except:
                return f"<set_len_{len(obj) if hasattr(obj, '__len__') else 'unknown'}_excluded>"
        
        # Handle complex objects by converting to string
        try:
            # Try to convert to string representation
            return str(obj)
        except:
            return f"<object_type_{type(obj).__name__}_excluded>"
    
    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load pipeline checkpoint from storage.
        
        Returns:
            Checkpoint data dictionary if found, None otherwise
        """
        try:
            checkpoint_blob = f"{self.run_folder}/.checkpoint.json"
            
            if not self.storage_client.blob_exists(checkpoint_blob):
                logger.info(f"No checkpoint found at: {checkpoint_blob}")
                return None
            
            checkpoint_content = self.storage_client.download_as_string(checkpoint_blob)
            if not checkpoint_content:
                logger.error(f"Failed to download checkpoint: {checkpoint_blob}")
                return None
            
            checkpoint_data = json.loads(checkpoint_content)
            
            # Validate checkpoint compatibility
            if not self._validate_checkpoint(checkpoint_data):
                logger.error("Checkpoint validation failed")
                return None
            
            logger.info(f"Loaded checkpoint with {len(checkpoint_data.get('completed_steps', []))} completed steps")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        Validate checkpoint compatibility with current pipeline.
        
        Args:
            checkpoint_data: Checkpoint data to validate
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ["pipeline_name", "run_guid", "completed_steps", "context"]
            for field in required_fields:
                if field not in checkpoint_data:
                    logger.error(f"Checkpoint missing required field: {field}")
                    return False
            
            # Check pipeline name matches
            if checkpoint_data["pipeline_name"] != self.pipeline_name:
                logger.error(f"Checkpoint pipeline name mismatch: {checkpoint_data['pipeline_name']} != {self.pipeline_name}")
                return False
            
            # Check run_guid matches
            if checkpoint_data["run_guid"] != self.run_guid:
                logger.error(f"Checkpoint run_guid mismatch: {checkpoint_data['run_guid']} != {self.run_guid}")
                return False
            
            # Check that completed steps exist in current pipeline
            completed_steps = checkpoint_data["completed_steps"]
            for step_name in completed_steps:
                if step_name not in self.steps:
                    logger.error(f"Checkpoint references unknown step: {step_name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating checkpoint: {e}")
            return False
    
    def _restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore pipeline state from checkpoint.
        
        Args:
            checkpoint_data: Checkpoint data to restore from
            
        Returns:
            Restored context dictionary
        """
        try:
            completed_steps = checkpoint_data["completed_steps"]
            context = checkpoint_data["context"]
            steps_summary = checkpoint_data.get("steps_summary", {})
            
            # Restore step states
            for step_name in completed_steps:
                if step_name in self.steps:
                    step = self.steps[step_name]
                    step.status = StepStatus.COMPLETED
                    
                    # Restore step metadata if available
                    if step_name in steps_summary:
                        step_data = steps_summary[step_name]
                        if "start_time" in step_data:
                            step.start_time = datetime.fromisoformat(step_data["start_time"])
                        if "end_time" in step_data:
                            step.end_time = datetime.fromisoformat(step_data["end_time"])
                        if "output_path" in step_data:
                            step.output_path = step_data["output_path"]
                        if "metadata" in step_data:
                            step.metadata = step_data["metadata"]
            
            logger.info(f"Restored pipeline state from checkpoint. Completed steps: {len(completed_steps)}")
            return context
            
        except Exception as e:
            logger.error(f"Error restoring from checkpoint: {e}")
            raise
    
    def _cleanup_checkpoint(self) -> bool:
        """
        Clean up checkpoint file after successful pipeline completion.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            checkpoint_blob = f"{self.run_folder}/.checkpoint.json"
            
            if self.storage_client.blob_exists(checkpoint_blob):
                success = self.storage_client.delete_blob(checkpoint_blob)
                if success:
                    logger.info(f"Cleaned up checkpoint file: {checkpoint_blob}")
                else:
                    logger.warning(f"Failed to clean up checkpoint file: {checkpoint_blob}")
                return success
            
            return True  # No checkpoint to clean up
            
        except Exception as e:
            logger.error(f"Error cleaning up checkpoint: {e}")
            return False
    
    def _get_failed_steps(self) -> List[str]:
        """Get list of step names that failed."""
        return [name for name, step in self.steps.items() if step.status == StepStatus.ERROR]
    
    def _get_completed_steps(self) -> List[str]:
        """Get list of step names that completed successfully."""
        return [name for name, step in self.steps.items() if step.status == StepStatus.COMPLETED]
    
    def _get_skipped_steps(self) -> List[str]:
        """Get list of step names that were skipped."""
        return [name for name, step in self.steps.items() if step.status == StepStatus.SKIPPED]
    
    def run(self, context: Optional[Dict[str, Any]] = None, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """
        Execute the pipeline by running all steps in order.
        
        Args:
            context: Optional context dictionary to pass between steps
            resume_from_checkpoint: If True, attempt to resume from existing checkpoint
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Starting {self.pipeline_name} pipeline execution")
        self.status = PipelineStatus.RUNNING
        
        if context is None:
            context = {}
        
        # Handle checkpoint restoration
        completed_steps = []
        if resume_from_checkpoint:
            checkpoint_data = self._load_checkpoint()
            if checkpoint_data:
                context = self._restore_from_checkpoint(checkpoint_data)
                completed_steps = checkpoint_data["completed_steps"]
                logger.info(f"Resuming from checkpoint. Skipping {len(completed_steps)} completed steps")
            else:
                logger.info("No valid checkpoint found, starting from beginning")
        
        results = {
            "pipeline_name": self.pipeline_name,
            "status": PipelineStatus.RUNNING.value,
            "run_guid": self.run_guid,
            "run_folder": self.run_folder,
            "steps_completed": len(completed_steps),
            "steps_failed": 0,
            "steps_skipped": 0,
            "errors": [],
            "context": context,
            "resumed_from_checkpoint": resume_from_checkpoint and len(completed_steps) > 0
        }
        
        try:
            # Create run folder
            self._create_run_folder()
            
            # Execute each step in order
            for step_name, step_config in self.step_definitions.items():
                try:
                    # Skip if step already completed (from checkpoint)
                    if step_name in completed_steps:
                        self._skip_step(step_name, "Already completed (from checkpoint)")
                        results["steps_skipped"] += 1
                        continue
                    
                    step_function = step_config["function"]
                    
                    # Execute the step
                    step_result = self._execute_step(step_name, step_function, context)
                    
                    # Update context with step result if it's a dict
                    if isinstance(step_result, dict):
                        context.update(step_result)
                    else:
                        context[f"{step_name}_result"] = step_result
                    
                    # Save intermediate results if enabled
                    if self.save_intermediate:
                        self._save_step_output(step_name, step_result, filename=f"{step_name}_result.json")
                    
                    results["steps_completed"] += 1
                    completed_steps.append(step_name)
                    
                    # Save checkpoint after each completed step
                    if not self._save_checkpoint(context, completed_steps):
                        logger.warning(f"Failed to save checkpoint after step: {step_name}")
                    
                except Exception as e:
                    results["steps_failed"] += 1
                    error_msg = f"Step '{step_name}' failed: {str(e)}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)
                    
                    # Save checkpoint even on failure to preserve progress
                    if not self._save_checkpoint(context, completed_steps):
                        logger.warning(f"Failed to save checkpoint after step failure: {step_name}")
                    
                    # Decide whether to continue or stop
                    if self._should_stop_on_error(step_name, step_config):
                        break
            
            # Update final status
            if results["steps_failed"] > 0:
                self.status = PipelineStatus.ERROR
                results["status"] = PipelineStatus.ERROR.value
            else:
                self.status = PipelineStatus.COMPLETED
                results["status"] = PipelineStatus.COMPLETED.value
                
                # Clean up checkpoint file on successful completion
                self._cleanup_checkpoint()
            
            # Add pipeline summary
            results["pipeline_summary"] = self._get_pipeline_summary()
            results["context"] = context
            
            logger.info(f"Pipeline completed. Steps completed: {results['steps_completed']}, Failed: {results['steps_failed']}")
            return results
            
        except Exception as e:
            self.status = PipelineStatus.ERROR
            results["status"] = PipelineStatus.ERROR.value
            results["errors"].append(str(e))
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _should_stop_on_error(self, step_name: str, step_config: Dict[str, Any]) -> bool:
        """
        Determine if pipeline should stop on step error.
        
        Args:
            step_name: Name of the failed step
            step_config: Configuration of the failed step
            
        Returns:
            True if pipeline should stop, False to continue
        """
        # By default, stop on error unless step is marked as optional
        return not step_config.get("optional", False)
    
    def __str__(self) -> str:
        """String representation of the pipeline."""
        return f"{self.pipeline_name} Pipeline (Run: {self.run_guid[:8]}...)"
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Pipeline(name='{self.pipeline_name}', status={self.status.value}, run_guid='{self.run_guid}')"
