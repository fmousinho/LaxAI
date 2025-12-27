"""
Player identity management layer for associating ByteTrack tracks with persistent players.
"""

from .player import Player, PlayerState, TrackData
from .player_manager import PlayerManager
from .config import PlayerConfig
from .offline_config import OfflinePlayerConfig
from .offline_associator import OfflinePlayerAssociator, TrackInfo, PlayerInfo

__all__ = [
    'Player',
    'PlayerState',
    'TrackData',
    'PlayerManager',
    'PlayerConfig',
    'OfflinePlayerConfig',
    'OfflinePlayerAssociator',
    'TrackInfo',
    'PlayerInfo',
]
