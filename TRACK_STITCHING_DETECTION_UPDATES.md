# Track Stitching with Detection Updates

## Overview

The track stitching algorithm has been enhanced to properly update `multi_frame_detections` when tracks are stitched together. This ensures that the detection data remains consistent with the new track IDs created during the stitching process.

## Key Changes Made

### 1. Enhanced `stitch_tracks()` Function

**New Signature:**
```python
def stitch_tracks(
    tracks_data: Dict[int, TrackData],
    multi_frame_detections: List,
    similarity_threshold: float = track_stitching_config.stitch_similarity_threshold,
    max_time_gap: int = track_stitching_config.max_time_gap,
    appearance_weight: float = track_stitching_config.appearance_weight,
    temporal_weight: float = track_stitching_config.temporal_weight,
    motion_weight: float = track_stitching_config.motion_weight
) -> Tuple[Dict[int, TrackData], List]:
```

**Key Enhancements:**
- Now accepts `multi_frame_detections` as an input parameter
- Returns both stitched tracks and updated detections
- Maintains a `track_id_mapping` dictionary to map old IDs to new IDs
- Updates all detection frames with the new track IDs

### 2. Detection Update Logic

```python
# Update multi_frame_detections with new track IDs
logger.info("Updating multi_frame_detections with stitched track IDs")
updated_detections = []

for frame_detections in multi_frame_detections:
    if frame_detections.tracker_id is not None:
        # Create a copy of the detections for this frame
        updated_frame_detections = frame_detections
        
        # Update tracker IDs based on the mapping
        new_tracker_ids = []
        for old_id in frame_detections.tracker_id:
            new_id = track_id_mapping.get(old_id, old_id)
            new_tracker_ids.append(new_id)
        
        # Update the tracker_id array
        updated_frame_detections.tracker_id = np.array(new_tracker_ids)
        updated_detections.append(updated_frame_detections)
    else:
        # No tracker IDs to update
        updated_detections.append(frame_detections)
```

### 3. Enhanced `associate_tracks_to_players_with_stitching()` Function

**New Signature:**
```python
def associate_tracks_to_players_with_stitching(
    tracks_data: Dict[int, TrackData],
    multi_frame_detections: List,
    similarity_threshold: float = player_config.reid_similarity_threshold,
    stitch_similarity_threshold: float = track_stitching_config.stitch_similarity_threshold,
    max_time_gap: int = track_stitching_config.max_time_gap,
    appearance_weight: float = track_stitching_config.appearance_weight,
    temporal_weight: float = track_stitching_config.temporal_weight,
    motion_weight: float = track_stitching_config.motion_weight
) -> Tuple[set[Player], Dict[int, Player], List]:
```

**Key Changes:**
- Now accepts `multi_frame_detections` as input
- Returns updated detections along with players and track mappings
- Handles the case when stitching is disabled by returning original detections

### 4. Updated Application Integration

**In `application.py`:**
```python
players, track_to_player, multi_frame_detections = associate_tracks_to_players_with_stitching(
    tracks_data, multi_frame_detections
)
```

The application now receives the updated `multi_frame_detections` with the new track IDs, ensuring consistency throughout the pipeline.

## Benefits

1. **Data Consistency**: All detection data now uses the same track IDs as the stitched tracks
2. **Seamless Integration**: Video output and reporting use the correct track IDs
3. **Backwards Compatibility**: When stitching is disabled, original detections are returned unchanged
4. **Proper Mapping**: All original track IDs are properly mapped to their stitched counterparts

## Usage Example

```python
# Before stitching: tracks 1, 2, 3 might be stitched into track 1001
# After stitching: all detections previously labeled as tracks 1, 2, 3 
# are now labeled as track 1001 in multi_frame_detections

tracks_data = get_tracks_data()
multi_frame_detections = get_detections()

# Perform track stitching with detection updates
players, track_to_player, updated_detections = associate_tracks_to_players_with_stitching(
    tracks_data, multi_frame_detections
)

# updated_detections now contains the new track IDs
# All downstream processing uses consistent track IDs
```

This enhancement ensures that the track stitching algorithm provides complete and consistent updates to all parts of the detection and tracking pipeline.
