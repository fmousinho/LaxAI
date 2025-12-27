"""
Post-processing script to associate ByteTrack tracks with persistent player IDs.

Usage:
    python -m scripts.associate_players \\
        --tracks tracks.json \\
        --embeddings embeddings.npz \\
        --output player_tracks.json \\
        --similarity-threshold 0.75 \\
        --max-gap 150
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

from player.player_manager import PlayerManager
from player.player import TrackData
from player.config import PlayerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_tracks(tracks_path: Path) -> Dict[int, List[dict]]:
    """
    Load tracks from JSON file.
    
    Expected format:
    {
        "frames": [
            {
                "frame_id": 0,
                "objects": [
                    {"track_id": 1, "bbox": [x1, y1, x2, y2], "confidence": 0.9}
                ]
            }
        ]
    }
    
    Returns:
        Dictionary mapping frame_id -> list of track dicts
    """
    logger.info(f"Loading tracks from {tracks_path}")
    
    with open(tracks_path, 'r') as f:
        data = json.load(f)
    
    tracks_by_frame = {}
    
    for frame_data in data.get('frames', []):
        frame_id = frame_data['frame_id']
        objects = frame_data.get('objects', [])
        tracks_by_frame[frame_id] = objects
    
    logger.info(f"Loaded {len(tracks_by_frame)} frames with {sum(len(t) for t in tracks_by_frame.values())} total tracks")
    
    return tracks_by_frame


def load_embeddings(embeddings_path: Path) -> Dict[int, np.ndarray]:
    """
    Load track embeddings from .npz file.
    
    Returns:
        Dictionary mapping track_id -> embedding vector
    """
    if not embeddings_path.exists():
        logger.warning(f"Embeddings file not found: {embeddings_path}")
        return {}
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    
    data = np.load(embeddings_path, allow_pickle=True)
    
    embeddings = {}
    for key in data.files:
        if key.startswith('track_'):
            track_id = int(key.split('_')[1])
            embedding_data = data[key].item()
            
            # Use mean embedding
            if 'mean' in embedding_data:
                embeddings[track_id] = embedding_data['mean']
            elif isinstance(embedding_data, np.ndarray):
                embeddings[track_id] = embedding_data
    
    logger.info(f"Loaded embeddings for {len(embeddings)} tracks")
    
    return embeddings


def associate_players(
    tracks_by_frame: Dict[int, List[dict]],
    embeddings: Dict[int, np.ndarray],
    config: PlayerConfig
) -> PlayerManager:
    """
    Run player association across all frames.
    
    Args:
        tracks_by_frame: Tracks organized by frame
        embeddings: Track embeddings
        config: Player configuration
        
    Returns:
        PlayerManager with all associations
    """
    manager = PlayerManager(config)
    
    # Process frames in order
    for frame_id in sorted(tracks_by_frame.keys()):
        frame_tracks = []
        
        for track_dict in tracks_by_frame[frame_id]:
            track_id = track_dict['track_id']
            bbox = track_dict.get('bbox', track_dict.get('tlbr', [0, 0, 0, 0]))
            confidence = track_dict.get('confidence', 1.0)
            
            # Get embedding if available
            embedding = embeddings.get(track_id)
            
            track_data = TrackData(
                track_id=track_id,
                frame_id=frame_id,
                bbox=tuple(bbox),
                confidence=confidence,
                embedding=embedding
            )
            frame_tracks.append(track_data)
        
        manager.process_frame(frame_id, frame_tracks)
        
        if frame_id % 100 == 0:
            stats = manager.get_statistics()
            logger.info(
                f"Frame {frame_id}: {stats['total_players']} players "
                f"({stats['active_players']} active, {stats['inactive_players']} inactive)"
            )
    
    return manager


def save_player_tracks(manager: PlayerManager, output_path: Path) -> None:
    """Save player associations to JSON file."""
    logger.info(f"Saving player associations to {output_path}")
    
    data = manager.export_players()
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    stats = data['statistics']
    logger.info(
        f"Saved {stats['total_players']} players "
        f"({stats['total_tracks']} total track instances)"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Associate ByteTrack tracks with persistent player IDs'
    )
    
    parser.add_argument(
        '--tracks',
        type=Path,
        required=True,
        help='Path to tracks.json from ByteTrack'
    )
    
    parser.add_argument(
        '--embeddings',
        type=Path,
        help='Path to embeddings.npz file (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for player_tracks.json'
    )
    
    # Configuration options
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.7,
        help='Minimum cosine similarity for matching (default: 0.7)'
    )
    
    parser.add_argument(
        '--max-gap',
        type=int,
        default=150,
        help='Maximum frames between tracks to allow matching (default: 150)'
    )
    
    parser.add_argument(
        '--min-gap',
        type=int,
        default=5,
        help='Minimum frames before re-association (default: 5)'
    )
    
    parser.add_argument(
        '--ema-alpha',
        type=float,
        default=0.3,
        help='EMA alpha for appearance updates (default: 0.3)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger('player').setLevel(logging.DEBUG)
    
    # Load data
    tracks_by_frame = load_tracks(args.tracks)
    
    embeddings = {}
    if args.embeddings:
        embeddings = load_embeddings(args.embeddings)
    
    # Configure player management
    config = PlayerConfig(
        similarity_threshold=args.similarity_threshold,
        max_inactive_gap=args.max_gap,
        min_inactive_gap=args.min_gap,
        embedding_ema_alpha=args.ema_alpha,
        verbose=args.verbose,
    )
    
    # Run association
    logger.info("Starting player association...")
    manager = associate_players(tracks_by_frame, embeddings, config)
    
    # Save results
    save_player_tracks(manager, args.output)
    
    # Print final statistics
    stats = manager.get_statistics()
    logger.info("=" * 60)
    logger.info("Final Statistics:")
    logger.info(f"  Total players: {stats['total_players']}")
    logger.info(f"  Active players: {stats['active_players']}")
    logger.info(f"  Inactive players: {stats['inactive_players']}")
    logger.info(f"  Total track instances: {stats['total_tracks']}")
    logger.info(f"  Final frame: {stats['current_frame']}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
