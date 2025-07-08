"""
LaxAI Modules Package

This package contains the core modules for the LaxAI lacrosse video analysis system.
"""

from .player_association import (
    associate_tracks_to_players_greedy,
    associate_tracks_to_players_globally,
    cosine_similarity,
    find_optimal_batch_assignment,
    find_global_optimal_associations
)

__all__ = [
    'associate_tracks_to_players_greedy',
    'associate_tracks_to_players_globally', 
    'cosine_similarity',
    'find_optimal_batch_assignment',
    'find_global_optimal_associations'
]