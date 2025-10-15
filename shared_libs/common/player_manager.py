import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import shared_libs.config.logging_config

from supervision import Detections

logger = logging.getLogger(__name__)


class Player:
    """
    Represents a player with basic identification and track associations.

    Attributes:
        id: A unique identifier for the player (auto-generated)
        name: Optional name of the player
        track_ids: List of track IDs associated with this player
    """

    def __init__(self, player_id: int, name: Optional[str] = None, track_ids: Optional[List[int]] = None):
        """
        Initialize a Player instance.

        Args:
            player_id: Unique identifier for the player
            name: Optional name for the player
            track_ids: Optional list of associated track IDs
        """
        self.id = player_id
        self.name = name
        self.track_ids = track_ids if track_ids is not None else []

    def to_dict(self) -> Dict:
        """Convert player to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "track_ids": self.track_ids
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Player":
        """Create player from dictionary."""
        return cls(
            player_id=data["id"],
            name=data["name"],
            track_ids=data["track_ids"]
        )

    def __repr__(self) -> str:
        """Return a string representation of the Player."""
        return f"Player(id={self.id}, name={self.name}, track_ids={self.track_ids})"


class PlayerManager:
    """
    Manages players for a specific video session with JSON persistence.

    Maintains both player objects and track-to-player mappings for efficient lookups.
    Thread-safe for concurrent operations.
    """

    def __init__(self, video_id: str):
        """
        Initialize PlayerManager for a video.

        Args:
            video_id: Unique identifier for the video
        """
        self.video_id = video_id
        self.players: Dict[int, Player] = {}
        self.track_to_player: Dict[int, int] = {}  # track_id -> player_id
        self._next_player_id = 1
        self._lock = threading.Lock()

    def initialize(self, init_detections: Detections) -> None:
        """Create initial players from detections if none exist."""
        with self._lock:
            if self.players:
                logger.info(f"PlayerManager for video {self.video_id} already initialized with players.")
                return

            track_ids = init_detections.data.get("track_id", [])
            unique_track_ids = set(track_ids) if track_ids is not None else set()

            for track_id in unique_track_ids:
                new_player = self.create_player()
                self.add_track_to_player(new_player.id, track_id)

            logger.info(f"Initialized PlayerManager for video {self.video_id} with {len(self.players)} players.")

    def create_player(self, name: Optional[str] = None) -> Player:
        """
        Create a new player.

        Args:
            name: Optional name for the player

        Returns:
            The newly created Player instance
        """
        with self._lock:
            player_id = self._next_player_id
            self._next_player_id += 1

            player = Player(player_id=player_id, name=name)
            self.players[player_id] = player

            logger.info(f"Created player {player_id} for video {self.video_id}")
            return player

    def delete_player(self, player_id: int) -> bool:
        """
        Delete a player and remove all its track associations.

        Args:
            player_id: ID of the player to delete

        Returns:
            True if player was deleted, False if not found
        """
        with self._lock:
            if player_id not in self.players:
                logger.warning(f"Player {player_id} not found for deletion")
                return False

            player = self.players[player_id]

            # Remove track associations
            for track_id in player.track_ids:
                self.track_to_player.pop(track_id, None)

            # Remove player
            del self.players[player_id]

            logger.info(f"Deleted player {player_id} from video {self.video_id}")
            return True

    def update_player_name(self, player_id: int, name: Optional[str]) -> bool:
        """
        Update a player's name.

        Args:
            player_id: ID of the player to update
            name: New name for the player

        Returns:
            True if updated, False if player not found
        """
        with self._lock:
            if player_id not in self.players:
                logger.warning(f"Player {player_id} not found for name update")
                return False

            self.players[player_id].name = name
            logger.info(f"Updated name for player {player_id} to '{name}'")
            return True

    def add_track_to_player(self, player_id: int, track_id: int) -> bool:
        """
        Associate a track ID with a player.

        Args:
            player_id: ID of the player
            track_id: Track ID to associate

        Returns:
            True if added, False if player not found or track already associated
        """
        with self._lock:
            if player_id not in self.players:
                logger.warning(f"Player {player_id} not found for track association")
                return False

            player = self.players[player_id]
            if track_id in player.track_ids:
                logger.warning(f"Track {track_id} already associated with player {player_id}")
                return False

            # Check if track is already associated with another player
            if track_id in self.track_to_player:
                existing_player_id = self.track_to_player[track_id]
                if existing_player_id != player_id:
                    logger.warning(f"Track {track_id} already associated with player {existing_player_id}")
                    return False

            player.track_ids.append(track_id)
            self.track_to_player[track_id] = player_id

            logger.info(f"Associated track {track_id} with player {player_id}")
            return True

    def remove_track_from_player(self, player_id: int, track_id: int) -> bool:
        """
        Remove a track ID association from a player.

        Args:
            player_id: ID of the player
            track_id: Track ID to remove

        Returns:
            True if removed, False if not found or not associated
        """
        with self._lock:
            if player_id not in self.players:
                logger.warning(f"Player {player_id} not found for track removal")
                return False

            player = self.players[player_id]
            if track_id not in player.track_ids:
                logger.warning(f"Track {track_id} not associated with player {player_id}")
                return False

            player.track_ids.remove(track_id)
            self.track_to_player.pop(track_id, None)

            logger.info(f"Removed track {track_id} from player {player_id}")
            return True

    def get_player_by_id(self, player_id: int) -> Optional[Player]:
        """
        Get a player by ID.

        Args:
            player_id: Player ID to retrieve

        Returns:
            Player instance or None if not found
        """
        with self._lock:
            return self.players.get(player_id)

    def get_player_by_track_id(self, track_id: int) -> Optional[Player]:
        """
        Get a player by track ID.

        Args:
            track_id: Track ID to look up

        Returns:
            Player instance or None if track not associated
        """
        with self._lock:
            player_id = self.track_to_player.get(track_id)
            if player_id is not None:
                return self.players.get(player_id)
            return None

    def get_all_players(self) -> List[Player]:
        """
        Get all players.

        Returns:
            List of all Player instances
        """
        with self._lock:
            return list(self.players.values())

    def serialize_to_save(self) -> str:
        """
        Save player data to JSON file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            data = {
                "video_id": self.video_id,
                "next_player_id": self._next_player_id,
                "players": [player.to_dict() for player in self.players.values()],
                "track_to_player": self.track_to_player
            }
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Failed to serialize player data for video {self.video_id}: {e}")
            return ""


    def load_from_disk(self) -> bool:
        """
        Load player data from JSON file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            file_path = self._get_player_file_path()
            if not file_path.exists():
                logger.info(f"No player data file found for video {self.video_id}")
                return True  # Not an error, just no data

            with open(file_path, 'r') as f:
                data = json.load(f)

            with self._lock:
                self.video_id = data["video_id"]
                self._next_player_id = data["next_player_id"]

                self.players = {}
                for player_data in data["players"]:
                    player = Player.from_dict(player_data)
                    self.players[player.id] = player

                self.track_to_player = {int(k): v for k, v in data["track_to_player"].items()}

            logger.info(f"Loaded player data for video {self.video_id} from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load player data for video {self.video_id}: {e}")
            return False
        
    def load_players_from_json(self, json_object: str):
        """
        Load player data from a JSON object.

        Args:
            data: JSON object containing player data

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with self._lock:
                data = json.loads(json_object)
                file_video_id = data["video_id"]
                if file_video_id != self.video_id:
                    raise ValueError(f"Video ID mismatch: expected {self.video_id}, found {file_video_id}")

                self._next_player_id = data["next_player_id"]

                self.players = {}
                for player_data in data["players"]:
                    player = Player.from_dict(player_data)
                    self.players[player.id] = player

                self.track_to_player = {int(k): v for k, v in data["track_to_player"].items()}

            logger.info(f"Loaded player data for video {self.video_id} from JSON")
            return 

        except Exception as e:
            raise RuntimeError(f"Failed to load player data for video {self.video_id} from JSON: {e}")


def initialize_player_manager(video_id: str, current_frame_id: int, frame_detections: Detections) -> PlayerManager:
    """
    Initialize a PlayerManager for a given video ID, loading existing data if available.

    Args:
        video_id: Unique identifier for the video
        current_frame_id: The current frame ID being processed
        frame_detections: Detections for the current frame

    Returns:
        Initialized PlayerManager instance
    """
    try:
        manager = PlayerManager(video_id=video_id)
        manager.initialize(frame_detections)
        return manager
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PlayerManager for video {video_id}: {e}")
    

def load_player_manager(video_id: str, players_json: str) -> Optional[PlayerManager]:
    """
    Load a PlayerManager from a JSON object.

    Args:
        video_id: Unique identifier for the video
        players_json: JSON object containing player data

    Returns:
        Loaded PlayerManager instance
    """
    try:                        
        manager = PlayerManager(video_id)
        manager.load_players_from_json(players_json)
        return manager
    
    except Exception as e:
        logger.error (f"Failed to load PlayerManager for video {video_id}: {e}")
        return None
