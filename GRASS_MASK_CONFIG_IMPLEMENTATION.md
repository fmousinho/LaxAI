# Grass Mask Configuration Implementation

## Summary

Added configurable grass mask functionality to the DataPrepPipeline that allows enabling or disabling grass mask detection and background removal steps.

## Changes Made

### 1. Updated DataPrepPipeline.__init__()

- Added `enable_grass_mask` parameter (default: None)
- When None, uses `transform_config.enable_background_removal` as default
- Conditionally initializes `BackgroundMaskDetector` only when enabled
- Adds appropriate logging for enabled/disabled state

### 2. Dynamic Pipeline Step Configuration

- Pipeline steps are now built dynamically based on `enable_grass_mask` setting
- When enabled: includes `calculate_grass_mask` and `remove_crop_background` steps
- When disabled: skips these steps entirely
- Maintains correct step order:
  - **Enabled**: `['import_videos', 'load_videos', 'extract_frames', 'calculate_grass_mask', 'detect_players', 'extract_crops', 'remove_crop_background', 'augment_crops', 'create_training_and_validation_sets']`
  - **Disabled**: `['import_videos', 'load_videos', 'extract_frames', 'detect_players', 'extract_crops', 'augment_crops', 'create_training_and_validation_sets']`

### 3. Updated Step Methods

#### _initialize_grass_mask()
- Added early exit with error if grass mask is disabled
- Prevents execution when step is accidentally called

#### _remove_crop_background()
- Added early exit with error if grass mask is disabled
- Prevents execution when step is accidentally called

#### _augment_crops()
- Updated to work with both original crops (grass mask disabled) and modified crops (grass mask enabled)
- Dynamically determines source crops based on `enable_grass_mask` setting
- Uses appropriate in-memory crop variables based on processing path

### 4. Helper Method Updates

#### _download_crops_for_augmentation()
- Updated to accept generic `source_crops_folder` parameter
- Works with both original and modified crop folders

## Usage

### Explicit Configuration
```python
# Enable grass mask
pipeline = DataPrepPipeline(config, enable_grass_mask=True)

# Disable grass mask
pipeline = DataPrepPipeline(config, enable_grass_mask=False)
```

### Default Configuration
```python
# Uses transform_config.enable_background_removal setting
pipeline = DataPrepPipeline(config)
```

### Global Configuration
```python
from src.config.all_config import transform_config

# Modify global default
transform_config.enable_background_removal = True
pipeline = DataPrepPipeline(config)  # Will use True
```

## Benefits

1. **Performance**: When disabled, skips computationally expensive background removal
2. **Flexibility**: Can be configured per pipeline instance or globally
3. **Backward Compatibility**: Default behavior controlled by existing config
4. **Safety**: Prevents accidental execution of disabled steps
5. **Memory Efficiency**: Doesn't initialize BackgroundMaskDetector when disabled

## Testing

Comprehensive testing verified:
- ✅ Grass mask enabled: correct steps included, detector initialized
- ✅ Grass mask disabled: steps excluded, detector not initialized  
- ✅ Default config: matches transform_config.enable_background_removal setting
- ✅ Step ordering: correct sequence maintained in both modes
- ✅ Error handling: disabled steps return appropriate errors if called

## Files Modified

- `core/train/dataprep_pipeline.py`: Main implementation
- `examples/grass_mask_config_example.py`: Usage examples (new)
