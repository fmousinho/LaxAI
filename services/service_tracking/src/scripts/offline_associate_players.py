#!/usr/bin/env python3
"""
Offline player association script.

Usage:
    python -m scripts.offline_associate_players \\
        --tracks tracks.json \\
        --embeddings embeddings.pt \\
        --output players.json \\
        --frame-width 1920 \\
        --frame-height 1080
"""
import argparse
import json
import logging
from pathlib import Path

from player.offline_associator import OfflinePlayerAssociator
from player.offline_config import OfflinePlayerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Offline player association with team clustering'
    )
    
    parser.add_argument(
        '--tracks', type=Path, required=True,
        help='Path to tracks.json'
    )
    parser.add_argument(
        '--embeddings', type=Path, required=True,
        help='Path to embeddings.pt'
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output path for players.json'
    )
    parser.add_argument(
        '--frame-width', type=int, default=1920,
        help='Video frame width (default: 1920)'
    )
    parser.add_argument(
        '--frame-height', type=int, default=1080,
        help='Video frame height (default: 1080)'
    )
    parser.add_argument(
        '--fps', type=float, default=30.0,
        help='Video FPS (default: 30)'
    )
    parser.add_argument(
        '--similarity-threshold', type=float, default=0.65,
        help='Minimum similarity for matching (default: 0.65)'
    )
    parser.add_argument(
        '--min-players', type=int, default=11,
        help='Min players per team (default: 11)'
    )
    parser.add_argument(
        '--max-players', type=int, default=22,
        help='Max players per team (default: 22)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configure
    config = OfflinePlayerConfig(
        fps=args.fps,
        similarity_threshold=args.similarity_threshold,
        min_players_per_team=args.min_players,
        max_players_per_team=args.max_players,
        verbose=args.verbose,
    )
    
    # Run
    associator = OfflinePlayerAssociator(config)
    associator.load_data(
        str(args.tracks),
        str(args.embeddings),
        frame_size=(args.frame_width, args.frame_height)
    )
    
    result = associator.run()
    associator.save(str(args.output))
    
    # Print summary
    stats = result['statistics']
    logger.info("=" * 60)
    logger.info("Results:")
    logger.info(f"  Teams: {stats['teams']}")
    logger.info(f"  Players: {stats['total_players']}")
    logger.info(f"  Tracks: {stats['total_tracks']}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
