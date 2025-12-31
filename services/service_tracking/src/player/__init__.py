"""
Offline player association - batch processing of all tracks after tracking completes.
"""

from player.associator import PlayerAssociator, TrackInfo, Player
from player.config import PlayerAssociatorConfig

__all__ = [
    'PlayerAssociatorConfig',
    'PlayerAssociator',
    'TrackInfo',    
    'Player',
]
