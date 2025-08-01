"""
LaxAI Modules Package

This package contains the core modules for the LaxAI lacrosse video analysis system.
"""

from .player_association import (
    associate_tracks_to_players_greedy,
    associate_tracks_to_players_globally,
    associate_tracks_to_players_with_stitching,
    cosine_similarity,
    find_optimal_batch_assignment,
    stitch_tracks,
    calculate_stitching_cost,
    create_representative_embedding,
    update_detection_metadata
)

from src.train.augmentation import (
    augment_images,
    test_augmentation
)

__all__ = [
    'associate_tracks_to_players_greedy',
    'associate_tracks_to_players_globally',
    'associate_tracks_to_players_with_stitching',
    'cosine_similarity',
    'find_optimal_batch_assignment',
    'stitch_tracks',
    'calculate_stitching_cost',
    'create_representative_embedding',
    'update_detection_metadata',
    'augment_images',
    'test_augmentation'
]