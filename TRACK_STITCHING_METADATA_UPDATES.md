# Track Stitching Metadata Updates Enhancement

## Overview

This document describes the enhancement to the track stitching algorithm that ensures all metadata references to tracker IDs are updated when tracks are stitched together. This maintains consistency across the entire detection pipeline.

## Problem Statement

When tracks are stitched together, the `tracker_id` array in the detections is updated, but other metadata fields that reference the old tracker IDs remain unchanged. This can lead to inconsistencies where:

1. The main `tracker_id` array contains new stitched IDs
2. The `data` dictionary contains references to old tracker IDs
3. The `metadata` dictionary contains references to old tracker IDs
4. Custom fields or analysis results become misaligned

## Solution

The `stitch_tracks()` function has been enhanced to perform comprehensive metadata updates:

### 1. Primary Tracker ID Updates
- Updates the main `tracker_id` array with new stitched IDs
- Maintains the existing ID mapping functionality

### 2. Data Dictionary Updates
- Scans all fields in the `data` dictionary for tracker ID references
- Updates fields with 'track' in their name (case-insensitive)
- Performs general checks for any integer values matching known track IDs
- Handles both single values and arrays/lists

### 3. Metadata Dictionary Updates
- Scans all fields in the `metadata` dictionary for tracker ID references
- Updates fields with 'track' in their name (case-insensitive)
- Performs general checks for any integer values matching known track IDs
- Handles both single values and arrays/lists

### 4. Comprehensive Coverage
- Handles numpy arrays, Python lists, and single integer values
- Preserves data types (numpy arrays remain numpy arrays)
- Provides detailed logging for debugging and monitoring

## Implementation Details

### Detection Update Logic

```python
# Update tracker_id array
new_tracker_ids = [track_id_mapping.get(old_id, old_id) for old_id in frame_detections.tracker_id]
updated_frame_detections.tracker_id = np.array(new_tracker_ids)

# Update data dictionary
for key, values in data.items():
    # Check explicit track fields
    if 'track' in key.lower():
        # Update all values in the field
        
    # Check for any integer values matching known track IDs
    elif contains_trackable_integers(values):
        # Update matching values
        
# Update metadata dictionary (same logic as data)
```

### Logging and Monitoring

The enhanced implementation provides comprehensive logging:
- Debug logs for each field updated
- Summary counts of metadata and data field updates
- Clear distinction between explicit track fields and general integer updates

## Benefits

1. **Data Consistency**: All references to tracker IDs are updated consistently
2. **Pipeline Integrity**: Downstream analysis using metadata remains accurate
3. **Debugging Support**: Detailed logging helps identify what's being updated
4. **Flexible Coverage**: Handles both explicit track fields and general integer references
5. **Type Preservation**: Maintains original data types and structures

## Usage

The metadata update functionality is automatically enabled when using the track stitching algorithm:

```python
# Enhanced function signature
stitched_tracks, updated_detections = stitch_tracks(
    tracks_data=tracks_data,
    multi_frame_detections=multi_frame_detections,
    # ... other parameters
)

# The updated_detections now have all metadata updated
```

## Configuration

Metadata updates are controlled by the same track stitching configuration:

```python
# Enable/disable track stitching (includes metadata updates)
track_stitching_config.enable_stitching = True

# All track stitching parameters apply to metadata updates
```

## Logging Output

Example logging output during metadata updates:

```
INFO - Updating multi_frame_detections with stitched track IDs
DEBUG - Updated data field 'track_history' with stitched track IDs
DEBUG - Updated metadata field 'primary_track_id' with stitched track ID (general check)
INFO - Updated 150 frames with new track IDs
INFO - Updated 3 metadata fields with stitched track IDs
INFO - Updated 12 data fields with stitched track IDs
```

## Performance Considerations

The metadata update process adds minimal overhead:
- Metadata dictionaries are typically small
- Updates are performed only when track stitching is enabled
- Logging can be adjusted via log levels for production use

## Future Enhancements

Potential future improvements:
1. Configuration options for which metadata fields to update
2. Custom field mapping for non-standard metadata structures
3. Validation checks to ensure metadata consistency
4. Support for nested metadata structures

## Testing

To verify metadata updates are working correctly:
1. Enable debug logging
2. Run track stitching on a test dataset
3. Check logs for metadata update messages
4. Verify that downstream analysis uses consistent track IDs

## Troubleshooting

Common issues and solutions:
- **No metadata updates logged**: Check if detections have `data` or `metadata` dictionaries
- **Partial updates**: Verify field names contain 'track' or values match known track IDs
- **Type errors**: Ensure custom metadata follows expected formats (integers, lists, arrays)
