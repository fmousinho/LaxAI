# Track Stitching Implementation Summary

This document summarizes the track stitching algorithm implemented in the `player_association.py` module.

## Overview

The track stitching algorithm addresses the problem of fragmented tracklets that occur when the same player is tracked across multiple disconnected segments due to occlusions, detector failures, or tracking errors. By merging these fragments, we create longer, more complete tracks that improve the quality of training data for re-identification models.

## Algorithm Components

### 1. Representative Embedding Generation

Each tracklet is represented by a single embedding vector created by:
- Extracting embeddings from all crops in the tracklet using the Re-ID model
- Averaging these embeddings to create a single representative vector
- L2 normalizing the result for consistent similarity calculations

### 2. Matching Cost Calculation

The cost of stitching two tracklets is calculated using three components:

**Appearance Cost (Re-ID Score)**:
- Cosine distance between representative embeddings
- Primary factor for determining track similarity
- Weight: 1.0 (default)

**Temporal Cost (Time Gap)**:
- Penalizes matches with large time gaps
- Calculated as: `time_gap * temporal_weight`
- Weight: 0.1 (default)

**Motion Cost (Optional)**:
- Uses motion prediction to estimate expected position
- Currently simplified but can be enhanced with Kalman filtering
- Weight: 0.05 (default)

### 3. Stitching Process

The algorithm uses a greedy approach:
1. Sort tracklets by start time
2. For each tracklet, find potential matches (later tracklets from same team)
3. Calculate matching costs for all potential pairs
4. Apply similarity and time gap thresholds
5. Greedily select the best matches and merge tracklets
6. Update representative embeddings after each merge

## Configuration Parameters

### TrackStitchingConfig

- `enable_stitching`: Enable/disable track stitching (default: True)
- `stitch_similarity_threshold`: Minimum similarity for stitching (default: 0.7)
- `max_time_gap`: Maximum frame gap between tracklets (default: 60)
- `appearance_weight`: Weight for appearance similarity (default: 1.0)
- `temporal_weight`: Weight for temporal gap penalty (default: 0.1)
- `motion_weight`: Weight for motion prediction (default: 0.05)

## Usage

The main function is `associate_tracks_to_players_with_stitching()` which:

1. **Step 1**: Stitch fragmented tracklets using the algorithm above
2. **Step 2**: Associate stitched tracks to players using global optimization
3. **Step 3**: Create mapping from original track IDs to final players

## Benefits

- **Improved Training Data**: Longer tracks provide better learning examples
- **Reduced Fragmentation**: Fewer disconnected segments improve model performance
- **Better Re-ID**: More complete appearance variations per player
- **Configurable**: Parameters can be tuned for different scenarios

## Integration

The algorithm is integrated into the main application workflow:
- Automatically applied during player association
- Can be disabled via configuration if needed
- Maintains compatibility with existing code

## Future Enhancements

- **Motion Prediction**: Implement proper Kalman filtering for bbox prediction
- **Appearance Modeling**: Use more sophisticated embedding aggregation
- **Temporal Modeling**: Consider track duration and activity patterns
- **Multi-stage Stitching**: Apply stitching at different similarity thresholds
