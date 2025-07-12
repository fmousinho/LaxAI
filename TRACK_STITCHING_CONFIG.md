# Track Stitching Configuration Parameters

This document explains all the configurable parameters for the track stitching algorithm.

## Configuration Location

All parameters are defined in `config/all_config.py` in the `TrackStitchingConfig` class:

```python
@dataclass
class TrackStitchingConfig:
    """Configuration for track stitching parameters."""
    enable_stitching: bool = True
    stitch_similarity_threshold: float = 0.7
    max_time_gap: int = 60
    appearance_weight: float = 1.0
    temporal_weight: float = 0.1
    motion_weight: float = 0.05
```

## Parameter Details

### `enable_stitching: bool = True`
- **Purpose**: Master switch to enable/disable track stitching
- **Default**: `True`
- **Usage**: If `False`, the algorithm falls back to standard global association without stitching
- **Recommended**: Keep `True` for improved tracking performance

### `stitch_similarity_threshold: float = 0.7`
- **Purpose**: Minimum cosine similarity required between tracklets for stitching
- **Range**: 0.0 to 1.0 (higher = more strict)
- **Default**: `0.7`
- **Tuning**: 
  - Lower values (0.5-0.6): More aggressive stitching, may connect different players
  - Higher values (0.8-0.9): More conservative, may miss valid connections
- **Recommended**: Start with 0.7 and adjust based on results

### `max_time_gap: int = 60`
- **Purpose**: Maximum frame gap allowed between tracklets for stitching
- **Units**: Frames
- **Default**: `60` frames (2 seconds at 30 FPS)
- **Tuning**:
  - Smaller values (20-40): Only connect tracks with brief gaps
  - Larger values (80-120): Connect tracks with longer occlusions
- **Recommended**: Adjust based on typical occlusion duration in your videos

### `appearance_weight: float = 1.0`
- **Purpose**: Weight for appearance similarity in the cost calculation
- **Default**: `1.0`
- **Usage**: Primary factor in determining track similarity
- **Tuning**: Generally keep at 1.0 as the baseline, adjust other weights relative to this
- **Recommended**: Leave at 1.0 unless you have specific reasons to change

### `temporal_weight: float = 0.1`
- **Purpose**: Weight for temporal gap penalty in the cost calculation
- **Default**: `0.1`
- **Usage**: Penalizes connections with large time gaps
- **Tuning**:
  - Lower values (0.05): Less penalty for time gaps
  - Higher values (0.2-0.3): More penalty for time gaps
- **Recommended**: Adjust based on your tracking scenario

### `motion_weight: float = 0.05`
- **Purpose**: Weight for motion prediction in the cost calculation
- **Default**: `0.05`
- **Usage**: Currently minimal impact (motion prediction is simplified)
- **Future**: Will become more important when motion prediction is enhanced
- **Recommended**: Keep low until motion prediction is fully implemented

## Usage Example

```python
from config.all_config import track_stitching_config

# Use default values
players, track_to_player = associate_tracks_to_players_with_stitching(tracks_data)

# Or override specific parameters
players, track_to_player = associate_tracks_to_players_with_stitching(
    tracks_data,
    stitch_similarity_threshold=0.8,  # More strict
    max_time_gap=40  # Shorter time gaps
)
```

## Tuning Guidelines

1. **Start with defaults**: The default values work well for most lacrosse scenarios
2. **Adjust similarity threshold**: Most impactful parameter for balancing precision vs recall
3. **Tune time gap**: Based on typical occlusion patterns in your videos
4. **Monitor results**: Check stitching statistics in logs to see reduction in track count
5. **Iterate**: Fine-tune based on visual inspection of results

## Performance Impact

- **Similarity threshold**: Higher values = fewer stitches = faster processing
- **Time gap**: Larger values = more potential matches = slower processing
- **Weights**: Minimal impact on processing speed

## Troubleshooting

- **Too many false connections**: Increase `stitch_similarity_threshold`
- **Missing valid connections**: Decrease `stitch_similarity_threshold` or increase `max_time_gap`
- **Poor performance**: Disable stitching temporarily with `enable_stitching: False`
- **Inconsistent results**: Check embedding quality and normalization consistency
