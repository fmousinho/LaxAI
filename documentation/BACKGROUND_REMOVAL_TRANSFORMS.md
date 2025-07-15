# Background Removal Transforms

This document explains how to use the new background removal transforms in the LaxAI project.

## Overview

The background removal transforms integrate the `BackgroundMaskDetector` into the existing torchvision transform pipeline, allowing you to automatically remove backgrounds from images during training and inference.

## Key Features

- **Seamless Integration**: Works with existing transform pipelines
- **Multiple Modes**: Supports all transform modes (training, inference, validation, opencv_safe)
- **Flexible Configuration**: Uses the centralized `BackgroundMaskConfig` system
- **Performance Optimized**: Minimal overhead for background removal

## Usage

### 1. Basic Usage

```python
from modules.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
from config.transforms import get_transforms
from config.all_config import BackgroundMaskConfig

# Create and train background detector
frame_generator = create_frame_generator_from_images(sample_images)
background_detector = BackgroundMaskDetector(frame_generator=frame_generator)

# Get transforms with background removal
training_transforms = get_transforms('training', background_detector=background_detector)
inference_transforms = get_transforms('inference', background_detector=background_detector)

# Use transforms normally
processed_image = training_transforms(pil_image)
```

### 2. Custom Background Configuration

```python
# Create custom background configuration
bg_config = BackgroundMaskConfig(
    sample_frames=5,
    std_dev_multiplier=1.2,
    replacement_color=(255, 255, 255),  # White background
    top_crop_ratio=0.3,
    bottom_crop_ratio=0.1,
    verbose=True
)

# Create detector with custom config
background_detector = BackgroundMaskDetector(
    frame_generator=frame_generator,
    config=bg_config
)
```

### 3. Batch Transform Creation

```python
from config.transforms import create_transforms_with_background_removal

# Create all transform types with background removal
all_transforms = create_transforms_with_background_removal(background_detector)

# Access specific transforms
training_transforms = all_transforms['training']
inference_transforms = all_transforms['inference']
opencv_safe_transforms = all_transforms['opencv_safe']
```

### 4. OpenCV-Style Input

```python
# For numpy arrays (OpenCV BGR format)
opencv_transforms = get_transforms('opencv_safe', background_detector=background_detector)
result = opencv_transforms(numpy_image)  # numpy array input

# For PIL Images
pil_transforms = get_transforms('training', background_detector=background_detector)
result = pil_transforms(pil_image)  # PIL Image input
```

## Available Transform Modes

| Mode | Description | Input Type | Use Case |
|------|-------------|------------|----------|
| `training` | Training with data augmentation | PIL Image | Model training |
| `inference` | Inference without augmentation | PIL Image | Model evaluation |
| `validation` | Same as inference | PIL Image | Model validation |
| `opencv_safe` | OpenCV BGR input handling | numpy array | OpenCV integration |
| `opencv_safe_training` | OpenCV BGR with augmentation | numpy array | OpenCV training |
| `tensor_to_pil` | Convert tensor to PIL | torch.Tensor | Visualization |

## Configuration Parameters

The background removal uses the `BackgroundMaskConfig` class with these parameters:

```python
@dataclass
class BackgroundMaskConfig:
    sample_frames: int = 5                    # Frames to analyze
    std_dev_multiplier: float = 1.0           # Detection sensitivity
    replacement_color: Tuple[int, int, int] = (255, 255, 255)  # White background
    top_crop_ratio: float = 0.5               # Remove top 50%
    bottom_crop_ratio: float = 0.1            # Remove bottom 10%
    verbose: bool = True                      # Progress info
    hsv_min_values: Tuple[int, int, int] = (0, 0, 0)      # HSV bounds
    hsv_max_values: Tuple[int, int, int] = (179, 255, 255)
    min_std_multiplier: float = 0.1           # Validation bounds
    max_std_multiplier: float = 3.0
```

## Performance Considerations

- Background removal adds ~150% overhead to transform time
- Most of the overhead comes from the HSV color space conversion
- Consider pre-processing images if using the same background detector repeatedly
- The background detector is thread-safe and can be reused across multiple transforms

## Integration with Training Pipeline

### DataLoader Integration

```python
from torch.utils.data import DataLoader

# Create dataset with background removal transforms
dataset = YourDataset(transform=get_transforms('training', background_detector=detector))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use normally in training loop
for batch_idx, (images, labels) in enumerate(dataloader):
    # Images already have background removed
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### Model Training

```python
# Training function with background removal
def train_with_background_removal(model, dataloader, criterion, optimizer, background_detector):
    model.train()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Images already processed by background removal transforms
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
```

## Best Practices

1. **Background Detector Training**: Use representative sample images for training the background detector
2. **Configuration Tuning**: Adjust `std_dev_multiplier` based on background color consistency
3. **Performance**: Pre-train the background detector once and reuse it
4. **Validation**: Compare results with and without background removal to verify effectiveness
5. **Debugging**: Enable `verbose=True` during development to monitor background detection

## Example: Complete Training Setup

```python
# Complete example of training with background removal
from modules.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
from config.transforms import get_transforms
from config.all_config import BackgroundMaskConfig
from torch.utils.data import DataLoader

# 1. Create background detector
sample_images = load_sample_images()  # Your sample images
frame_gen = create_frame_generator_from_images(sample_images)

bg_config = BackgroundMaskConfig(
    sample_frames=5,
    std_dev_multiplier=1.0,
    replacement_color=(255, 255, 255),
    verbose=False
)

background_detector = BackgroundMaskDetector(
    frame_generator=frame_gen,
    config=bg_config
)

# 2. Create transforms
train_transforms = get_transforms('training', background_detector=background_detector)
val_transforms = get_transforms('validation', background_detector=background_detector)

# 3. Create datasets
train_dataset = YourDataset(data_path='train', transform=train_transforms)
val_dataset = YourDataset(data_path='val', transform=val_transforms)

# 4. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. Train model
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for images, labels in train_loader:
        # Background removal already applied by transforms
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            # Calculate validation metrics
```

This integration provides a seamless way to incorporate background removal into your existing training pipeline while maintaining compatibility with all existing transform functionality.
