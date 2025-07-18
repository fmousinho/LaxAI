# Pipeline Checkpoint and Resume System

## Overview

The Pipeline checkpoint and resume system provides automatic recovery capabilities for long-running data processing pipelines. This system allows pipelines to be interrupted and resumed from where they left off, preventing data loss and reducing processing time.

## Key Features

### 1. Automatic Checkpoint Saving
- Checkpoints are automatically saved after each completed step
- Checkpoint data includes:
  - Pipeline state and context
  - Completed steps list
  - Step metadata and timing information
  - Pipeline configuration details

### 2. Resume from Interruption
- Pipelines can be resumed from the last completed step
- Completed steps are automatically skipped
- Context is restored from the checkpoint
- No duplicate processing occurs

### 3. Checkpoint Validation
- Validates checkpoint compatibility with current pipeline
- Checks pipeline name and run GUID matching
- Ensures step compatibility between checkpoint and current pipeline

### 4. Automatic Cleanup
- Checkpoint files are automatically cleaned up on successful completion
- Failed pipelines preserve checkpoints for debugging and recovery

## Implementation Details

### Base Pipeline Class

The checkpoint functionality is implemented in the base `Pipeline` class in `core/common/pipeline.py`:

```python
def run(self, context: Optional[Dict[str, Any]] = None, resume_from_checkpoint: bool = False) -> Dict[str, Any]:
    """
    Execute the pipeline with optional checkpoint resume.
    
    Args:
        context: Pipeline context dictionary
        resume_from_checkpoint: Enable checkpoint resume functionality
    """
```

### Key Methods

#### `_save_checkpoint(context, completed_steps)`
- Saves pipeline state to Google Storage
- Includes context, completed steps, and metadata
- Returns success/failure status

#### `_load_checkpoint()`
- Loads checkpoint data from Google Storage
- Validates checkpoint compatibility
- Returns checkpoint data or None

#### `_restore_from_checkpoint(checkpoint_data)`
- Restores pipeline state from checkpoint
- Updates step statuses and metadata
- Returns restored context

#### `_cleanup_checkpoint()`
- Removes checkpoint file on successful completion
- Called automatically when pipeline completes without errors

### Checkpoint File Structure

Checkpoints are saved as JSON files in the pipeline's run folder:

```json
{
  "pipeline_name": "train_pipeline",
  "run_guid": "uuid-string",
  "run_folder": "runs/run_uuid",
  "timestamp": "2025-01-20T10:30:00.000Z",
  "completed_steps": ["step1", "step2", "step3"],
  "context": {
    "video_path": "path/to/video.mp4",
    "video_guid": "video-uuid",
    "frames_data": [...],
    "crops_in_memory": {...}
  },
  "steps_summary": {
    "step1": {
      "status": "completed",
      "start_time": "2025-01-20T10:30:00.000Z",
      "end_time": "2025-01-20T10:31:00.000Z",
      "metadata": {...}
    }
  },
  "checkpoint_version": "1.0"
}
```

## Usage Examples

### DataPrepPipeline with Checkpoints

```python
from core.train.dataprep_pipeline import DataPrepPipeline
from config.all_config import DetectionConfig

# Create pipeline
config = DetectionConfig()
pipeline = DataPrepPipeline(config, tenant_id="tenant1")

# Run with checkpoint resume enabled
results = pipeline.run(
    video_path="gs://bucket/video.mp4",
    resume_from_checkpoint=True
)

# Check if resumed from checkpoint
if results.get('resumed_from_checkpoint'):
    print("Pipeline resumed from previous checkpoint")
```

### Custom Pipeline Implementation

```python
from core.common.pipeline import Pipeline

class CustomPipeline(Pipeline):
    def __init__(self):
        step_definitions = {
            "step1": {"function": self._step1, "description": "Process data"},
            "step2": {"function": self._step2, "description": "Transform data"},
            "step3": {"function": self._step3, "description": "Save results"}
        }
        
        super().__init__(
            pipeline_name="custom_pipeline",
            storage_client=storage_client,
            step_definitions=step_definitions
        )
    
    def run_with_checkpoints(self, data_path: str):
        context = {"data_path": data_path}
        return super().run(context, resume_from_checkpoint=True)
```

## Benefits

### 1. Fault Tolerance
- Automatic recovery from interruptions
- No data loss on system crashes
- Preserves expensive computations

### 2. Time Efficiency
- Skip completed steps on resume
- Reduce overall processing time
- Minimize resource usage

### 3. Development Productivity
- Debug specific pipeline steps
- Iterate on pipeline improvements
- Test individual components

### 4. In-Memory Optimization Compatibility
- Checkpoints preserve in-memory data structures
- `crops_in_memory` and `modified_crops_in_memory` are saved
- Maintains optimization benefits across restarts

## Storage Considerations

### Checkpoint Storage
- Checkpoints are stored in Google Cloud Storage
- Location: `{run_folder}/.checkpoint.json`
- Automatically managed by the pipeline system

### Context Size Management
- Large in-memory objects (like image data) are serialized
- JSON-compatible data structures are required
- Consider memory usage for large datasets

## Error Handling

### Checkpoint Save Failures
- Pipeline continues running even if checkpoint save fails
- Warning messages are logged
- Does not interrupt pipeline execution

### Checkpoint Load Failures
- Pipeline starts fresh if checkpoint cannot be loaded
- Validation errors are logged
- Graceful fallback to normal execution

### Validation Failures
- Incompatible checkpoints are rejected
- Pipeline starts from beginning
- Detailed error messages for debugging

## Best Practices

### 1. Enable Checkpoints for Long-Running Pipelines
```python
# Always enable for production pipelines
results = pipeline.run(
    video_path=video_path,
    resume_from_checkpoint=True
)
```

### 2. Use Unique Run GUIDs
- Each pipeline run should have a unique GUID
- Prevents checkpoint conflicts
- Enables parallel pipeline execution

### 3. Monitor Checkpoint Storage
- Check available storage space
- Monitor checkpoint file sizes
- Clean up old runs periodically

### 4. Test Resume Functionality
- Verify checkpoint compatibility
- Test interruption and resume scenarios
- Validate data integrity after resume

## Architecture Integration

### Pipeline Inheritance
- All pipelines inherit checkpoint functionality
- No additional implementation required
- Consistent behavior across all pipeline types

### Storage Integration
- Uses existing GoogleStorageClient
- Leverages pipeline's storage configuration
- Consistent with other pipeline artifacts

### Context Preservation
- Maintains pipeline context across restarts
- Preserves optimization data structures
- Ensures consistent pipeline behavior

## Performance Impact

### Checkpoint Overhead
- Minimal impact on pipeline execution time
- Asynchronous checkpoint saving
- Optimized JSON serialization

### Storage Efficiency
- Compressed checkpoint data
- Incremental checkpoint updates
- Automatic cleanup on completion

### Memory Usage
- Efficient context serialization
- Minimal memory overhead
- Optimized for large datasets

## Future Enhancements

### 1. Incremental Checkpoints
- Save only changed context data
- Reduce checkpoint file sizes
- Improve save performance

### 2. Checkpoint Compression
- Compress checkpoint files
- Reduce storage requirements
- Faster transfer times

### 3. Parallel Step Execution
- Support for parallel pipeline steps
- Distributed checkpoint management
- Enhanced scalability

### 4. Checkpoint Versioning
- Multiple checkpoint versions
- Rollback capabilities
- Version compatibility checks

## Troubleshooting

### Common Issues

#### Checkpoint Not Found
- Check run GUID matches
- Verify storage permissions
- Ensure checkpoint file exists

#### Validation Failures
- Verify pipeline name matches
- Check step compatibility
- Review checkpoint format

#### Context Restoration Errors
- Validate context data types
- Check serialization compatibility
- Review context size limits

### Debug Mode
Enable verbose logging to troubleshoot checkpoint issues:

```python
pipeline = DataPrepPipeline(
    config=config,
    tenant_id="tenant1",
    verbose=True  # Enable detailed logging
)
```

### Log Analysis
Monitor pipeline logs for checkpoint-related messages:
- Checkpoint save success/failure
- Checkpoint load attempts
- Validation results
- Step skip confirmations

## Conclusion

The checkpoint and resume system provides robust fault tolerance for data processing pipelines. It automatically handles interruptions, preserves progress, and ensures efficient resource utilization. The system is designed to be transparent to pipeline implementations while providing powerful recovery capabilities.

The integration with the existing optimization system (in-memory data passing) ensures that performance benefits are maintained even when using checkpoints, making it ideal for production environments where both reliability and performance are critical.
