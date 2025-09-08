# GPU Memory Management Utilities

This directory contains utilities for managing GPU memory in PyTorch training, specifically designed to handle CUDA Out of Memory (OOM) errors and memory cleanup.

## Files

### `gpu_memory.py`
Core GPU memory management functions:
- `clear_gpu_memory(force=False)`: Comprehensive GPU memory cleanup
- `get_gpu_memory_stats()`: Get current memory usage statistics  
- `log_gpu_memory_stats(prefix)`: Log memory stats with custom prefix
- `GPUMemoryContext`: Context manager for memory monitoring
- `reset_peak_memory_stats()`: Reset PyTorch's peak memory tracking

### `cleanup_gpu.py`
Standalone script for GPU memory cleanup:
```bash
# Show memory stats only
python src/utils/cleanup_gpu.py --stats-only

# Standard cleanup
python src/utils/cleanup_gpu.py

# Aggressive cleanup (searches for lingering tensors)
python src/utils/cleanup_gpu.py --force
```

## Integration with Training

The `Training` class now includes automatic memory management:

1. **Startup Cleanup**: Clears GPU memory on initialization to recover from previous crashes
2. **Memory Monitoring**: Logs memory usage at key training milestones
3. **OOM Error Handling**: Specific handling for CUDA Out of Memory errors with cleanup and recommendations
4. **Exception Cleanup**: Memory cleanup on any training failure

### Usage in Training

```python
# Memory is automatically cleared on initialization
trainer = Training(clear_memory_on_start=True)  # default

# Memory monitoring is automatic during training
trainer.train()  # Logs memory at epoch start and key points
```

## Memory Management Strategy

### Why GPU Memory "Sticks" at High Levels

PyTorch/CUDA uses a caching allocator that:
1. Allocates large memory chunks upfront
2. Reuses freed memory without returning it to the OS
3. Creates a "high water mark" effect

This is normal and efficient behavior - memory usage should stabilize, not continuously grow.

### When to Use Manual Cleanup

- **After OOM crashes**: Previous allocations may persist
- **Between training runs**: Ensure clean starting state
- **When switching models**: Clear memory from previous model
- **Development/debugging**: Reset to known clean state

### OOM Error Recovery

When CUDA runs out of memory:

1. **Automatic cleanup** is triggered with detailed logging
2. **Recommendations** are provided:
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training
   - Use smaller model architecture

3. **Manual recovery** options:
   ```bash
   # Quick cleanup
   python src/utils/cleanup_gpu.py
   
   # Aggressive cleanup
   python src/utils/cleanup_gpu.py --force
   ```

## Configuration

The training class accepts these memory-related parameters:

```python
Training(
    clear_memory_on_start=True,  # Clear memory on init
    batch_size=32,               # Primary memory control
    # ... other params
)
```

## Monitoring

Memory usage is logged at these points:
- Training initialization
- After model moves to GPU  
- After optimizer creation
- Start of each epoch
- During OOM error handling

Log messages include allocated and cached memory in GB for easy monitoring.
