# Centralized Background Removal Configuration

This document explains the new centralized background removal configuration system in the LaxAI project.

## Overview

The background removal system has been redesigned to use a centralized configuration approach instead of requiring callers to pass background detectors to individual transform functions. This provides a cleaner, more maintainable API with global configuration control.

## Key Improvements

### Before (Old API)
```python
# Every caller had to manage background detector
detector = BackgroundMaskDetector(frame_generator)
transforms = get_transforms('training', background_detector=detector)
```

### After (New API)
```python
# Global configuration - set once, use everywhere
transform_config.enable_background_removal = True
initialize_background_removal(sample_images)
refresh_transform_instances()

# All transforms now automatically use background removal
transforms = get_transforms('training')
```

## Configuration Parameters

### TransformConfig Class

The `TransformConfig` class in `config/all_config.py` now includes:

```python
@dataclass
class TransformConfig:
    # ... existing parameters ...
    
    # Background removal configuration
    enable_background_removal: bool = True  # Global enable/disable flag
    background_detector_sample_frames: int = 5  # Frames for detector training
    background_detector_auto_train: bool = True  # Auto-train detector
```

## Usage Guide

### 1. Basic Setup

```python
from config import (
    transform_config,
    initialize_background_removal,
    refresh_transform_instances,
    get_transforms
)

# Enable background removal
transform_config.enable_background_removal = True

# Initialize with sample images
sample_images = load_your_sample_images()  # List of RGB numpy arrays
initialize_background_removal(sample_images)

# Refresh transform instances to pick up changes
refresh_transform_instances()

# Now all transforms use background removal
training_transforms = get_transforms('training')
inference_transforms = get_transforms('inference')
```

### 2. Manual Detector Setup

```python
from config import (
    set_global_background_detector,
    create_background_detector_from_images,
    refresh_transform_instances
)

# Create detector manually
detector = create_background_detector_from_images(sample_images)

# Set as global detector
set_global_background_detector(detector)

# Enable background removal
transform_config.enable_background_removal = True
refresh_transform_instances()
```

### 3. Dynamic Configuration

```python
# Disable background removal
transform_config.enable_background_removal = False
refresh_transform_instances()

# Re-enable with new detector
transform_config.enable_background_removal = True
initialize_background_removal(new_sample_images)
refresh_transform_instances()
```

### 4. Check Current State

```python
from config import is_background_removal_enabled, get_global_background_detector

# Check if background removal is active
if is_background_removal_enabled():
    print("Background removal is enabled and detector is available")

# Get current detector
detector = get_global_background_detector()
```

## API Reference

### Configuration Functions

#### `initialize_background_removal(sample_images)`
Initialize background removal system with sample images.

**Parameters:**
- `sample_images`: List of RGB numpy arrays for background detector training

**Usage:**
```python
initialize_background_removal(sample_images)
```

#### `set_global_background_detector(detector)`
Set the global background detector instance.

**Parameters:**
- `detector`: BackgroundMaskDetector instance or None

**Usage:**
```python
set_global_background_detector(detector)
```

#### `get_global_background_detector()`
Get the current global background detector.

**Returns:**
- BackgroundMaskDetector instance or None

#### `refresh_transform_instances()`
Refresh all transform instances to pick up configuration changes.

**Usage:**
```python
refresh_transform_instances()
```

#### `is_background_removal_enabled()`
Check if background removal is enabled and available.

**Returns:**
- Boolean indicating if background removal is active

#### `create_background_detector_from_images(images)`
Create and train a background detector from sample images.

**Parameters:**
- `images`: List of RGB numpy arrays

**Returns:**
- Trained BackgroundMaskDetector instance

### Transform Functions

#### `get_transforms(mode, background_detector=None)`
Get transforms for the specified mode.

**Parameters:**
- `mode`: Transform mode ('training', 'inference', 'validation', 'opencv_safe', 'opencv_safe_training', 'tensor_to_pil')
- `background_detector`: **Deprecated** - use global configuration instead

**Returns:**
- transforms.Compose pipeline

**Usage:**
```python
# New way (recommended)
transforms = get_transforms('training')

# Old way (still supported for backward compatibility)
transforms = get_transforms('training', background_detector=detector)
```

## Migration Guide

### From Old API to New API

**Step 1: Update Configuration**
```python
# Old way
detector = BackgroundMaskDetector(frame_generator)
transforms = get_transforms('training', background_detector=detector)

# New way
transform_config.enable_background_removal = True
initialize_background_removal(sample_images)
refresh_transform_instances()
transforms = get_transforms('training')
```

**Step 2: Centralize Detector Management**
```python
# Old way - detector created everywhere
def create_dataset():
    detector = BackgroundMaskDetector(frame_generator)
    return Dataset(transform=get_transforms('training', background_detector=detector))

# New way - detector managed globally
def setup_background_removal():
    transform_config.enable_background_removal = True
    initialize_background_removal(sample_images)
    refresh_transform_instances()

def create_dataset():
    return Dataset(transform=get_transforms('training'))
```

## Configuration Examples

### Example 1: Training Pipeline

```python
from config import transform_config, initialize_background_removal, refresh_transform_instances, get_transforms

# Setup background removal
transform_config.enable_background_removal = True
transform_config.background_detector_sample_frames = 5
transform_config.background_detector_auto_train = True

# Initialize with sample images
sample_images = load_background_samples()
initialize_background_removal(sample_images)
refresh_transform_instances()

# Create datasets
train_dataset = Dataset(transform=get_transforms('training'))
val_dataset = Dataset(transform=get_transforms('validation'))

# Train model
for epoch in range(num_epochs):
    train_model(train_dataset)
    validate_model(val_dataset)
```

### Example 2: Inference Pipeline

```python
from config import transform_config, initialize_background_removal, refresh_transform_instances, get_transforms

# Setup for inference
transform_config.enable_background_removal = True
initialize_background_removal(background_samples)
refresh_transform_instances()

# Get inference transforms
inference_transforms = get_transforms('inference')

# Process images
for image in input_images:
    processed = inference_transforms(image)
    prediction = model(processed)
```

### Example 3: Dynamic Configuration

```python
from config import transform_config, refresh_transform_instances, get_transforms

# Start without background removal
transform_config.enable_background_removal = False
refresh_transform_instances()

# Process some images
initial_transforms = get_transforms('inference')
initial_results = process_images(images, initial_transforms)

# Enable background removal based on results
if background_needed:
    transform_config.enable_background_removal = True
    initialize_background_removal(sample_images)
    refresh_transform_instances()
    
    # Process with background removal
    bg_transforms = get_transforms('inference')
    bg_results = process_images(images, bg_transforms)
```

## Best Practices

1. **Global Setup**: Configure background removal once at application startup
2. **Refresh After Changes**: Always call `refresh_transform_instances()` after configuration changes
3. **Sample Images**: Use representative sample images for detector training
4. **Performance**: Enable background removal only when needed
5. **Testing**: Test both enabled and disabled states during development

## Performance Considerations

- Background removal adds ~27% overhead to transform time (improved from 376% in the old system)
- The global detector is shared across all transforms, reducing memory usage
- Use `is_background_removal_enabled()` to check state before processing

## Backward Compatibility

The old API is still supported for backward compatibility:

```python
# Old API (deprecated but still works)
detector = BackgroundMaskDetector(frame_generator)
transforms = get_transforms('training', background_detector=detector)

# New API (recommended)
transform_config.enable_background_removal = True
initialize_background_removal(sample_images)
refresh_transform_instances()
transforms = get_transforms('training')
```

## Error Handling

```python
from config import is_background_removal_enabled, get_global_background_detector

# Check if background removal is properly configured
if transform_config.enable_background_removal:
    if not is_background_removal_enabled():
        print("Warning: Background removal enabled but no detector available")
        print("Call initialize_background_removal() or set_global_background_detector()")
    
    detector = get_global_background_detector()
    if detector is None:
        print("No global background detector set")
```

## Summary

The new centralized background removal configuration system provides:

- **Cleaner API**: No need to pass detectors to every transform function
- **Global Control**: Enable/disable background removal system-wide
- **Better Performance**: Shared detector reduces overhead
- **Easier Maintenance**: Central configuration point
- **Backward Compatibility**: Old API still works
- **Dynamic Configuration**: Enable/disable at runtime

This system makes it much easier to manage background removal across your entire lacrosse video analysis pipeline while maintaining full flexibility and control.
