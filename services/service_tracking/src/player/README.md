# Player Identity Management

A layer on top of ByteTrack to associate multiple track IDs with persistent real-world players.

## Quick Start

```bash
# Run ByteTrack tracking (existing workflow)
python -m src.main --video video.mp4 --save-tracks tracks.json --save-embeddings embeddings.npz

# Associate tracks with players
python -m src.scripts.associate_players \\
    --tracks tracks.json \\
    --embeddings embeddings.npz \\
    --output player_tracks.json \\
    --similarity-threshold 0.75 \\
    --max-gap 150
```

## How It Works

**Tracks are ephemeral, Players are persistent.**

- A Player owns multiple track IDs over time (never overlapping)
- Each track belongs to exactly one player
- When tracks end/restart (occlusion, FOV exit), the same Player is maintained

### Algorithm

1. **Track Birth**: New track appears
   - Match to inactive players using appearance similarity (cosine distance on embeddings)
   - Enforce temporal constraints (no overlap, reasonable gap)
   - If matched → attach to existing Player
   - If no match → create new Player

2. **Track Continue**: Existing track updates
   - Update Player appearance model (EMA)
   - Track which player owns this track

3. **Track Death**: Track disappears
   - Mark Player as inactive
   - Store final appearance for future matching

## Configuration

```python
from player import PlayerConfig

config = PlayerConfig(
    similarity_threshold=0.7,      # Min cosine similarity to match
    max_inactive_gap=150,          # Max frames between track end and new match
    min_inactive_gap=5,            # Min frames to prevent flip-flopping
    embedding_ema_alpha=0.3,       # Weight for new embeddings in EMA
    min_embeddings_for_match=3,    # Min embeddings before matching
)
```

## Python API

```python
from player import PlayerManager, PlayerConfig, TrackData

# Initialize
config = PlayerConfig(similarity_threshold=0.75)
manager = PlayerManager(config)

# Process each frame
for frame_id, tracks in tracks_by_frame.items():
    track_data_list = [
        TrackData(
            track_id=t['track_id'],
            frame_id=frame_id,
            bbox=t['bbox'],
            confidence=t['confidence'],
            embedding=embeddings.get(t['track_id'])
        )
        for t in tracks
    ]
    
    manager.process_frame(frame_id, track_data_list)

# Export results
player_data = manager.export_players()
```

## Output Format

`player_tracks.json`:
```json
{
  "players": {
    "1": {
      "player_id": 1,
      "track_ids": [5, 23, 45],
      "track_segments": [
        [5, 0, 100],
        [23, 120, 200],
        [45, 230, 350]
      ],
      "first_seen_frame": 0,
      "last_seen_frame": 350,
      "state": "inactive",
      "embedding_count": 47
    }
  },
  "track_to_player": {
    "5": 1,
    "23": 1,
    "45": 1
  },
  "statistics": {
    "total_players": 15,
    "total_tracks": 45
  }
}
```

## Design Principles

✅ **Separation of Concerns**: ByteTrack unchanged, player logic separate  
✅ **Post-Processing**: Runs after tracking completes  
✅ **Appearance-Based**: Uses ReID embeddings for matching  
✅ **Temporal Constraints**: Prevents unrealistic associations  
✅ **Production-Ready**: Clean code, logging, configuration
