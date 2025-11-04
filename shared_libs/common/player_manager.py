import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional


from shared_libs.common.google_storage import GCSPaths
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
    
    # Define which attributes can be updated externally (track_ids have dedicated methods)
    UPDATABLE_ATTRIBUTES = {'name', 'image_path', 'player_number', 'team_id'}

    def __init__(
            self, 
            player_id: int, 
            name: Optional[str] = None, 
            track_ids: Optional[List[int]] = None,
            image_path: Optional[str] = None,
            player_number: Optional[int] = None,
            team_id: Optional[int] = None
    ):
        """
        Initialize a Player instance.

        Args:
            player_id: Unique identifier for the player
            name: Optional name for the player
            track_ids: Optional list of associated track IDs
            image_path: Optional path to the player's image
            player_number: Optional jersey number of the player (non-negative integer)
        """
        self.id = player_id
        self.name = name
        self.track_ids = track_ids if track_ids is not None else []
        self.image_path = image_path
        self.player_number = player_number
        self.team_id = team_id

    def to_dict(self) -> Dict:
        """Convert player to dictionary for JSON serialization."""
        # Convert numpy types to native Python types for JSON serialization
        return {
            "player_id": int(self.id) if self.id is not None else None,
            "player_name": self.name,
            "tracker_ids": [int(tid) for tid in self.track_ids],
            "image_path": self.image_path,
            "player_number": int(self.player_number) if self.player_number is not None else None,
            "team_id": int(self.team_id) if self.team_id is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Player":
        """Create player from dictionary."""
        # Support both old format (id) and new format (player_id) for backwards compatibility
        player_id = data.get("player_id", data.get("id"))
        player_name = data.get("player_name", data.get("name"))
        if player_id is None:
            raise ValueError("Player data must contain 'player_id' or 'id'")
        return cls(
            player_id=int(player_id),
            name=player_name,
            track_ids=data.get("tracker_ids", []),
            image_path=data.get("image_path"),
            player_number=data.get("player_number"),
            team_id=data.get("team_id"),
        )

    def __repr__(self) -> str:
        """Return a string representation of the Player."""
        return f"Player(id={self.id}, name={self.name}, number={self.player_number}, team_id={self.team_id}, track_ids={self.track_ids}, image_path={self.image_path})"


class PlayerManager:
    """
    Manages players for a specific video session with JSON persistence.

    Maintains both player objects and track-to-player mappings for efficient lookups.
    Thread-safe for concurrent operations.
    """
    path_manager = GCSPaths()

    def __init__(self, video_id: str, storage):
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
        self.storage = storage

    def initialize(self, init_detections: Detections) -> None:
        """Create initial players from detections if none exist."""
        with self._lock:
            if self.players:
                logger.info(f"PlayerManager for video {self.video_id} already initialized with players.")
                return
            logger.info(f"Analyzing detections to initialize PlayerManager for video {self.video_id}.")
            track_ids = init_detections.tracker_id
            unique_track_ids = set(track_ids) if track_ids is not None else set()

            for track_id in unique_track_ids:
                new_player = self._create_player()
                self._add_track_to_player(new_player.id, track_id)

            logger.info(f"Initialized PlayerManager for video {self.video_id} with {len(self.players)} players.")

    def create_player(self, name: Optional[str] = None, image_path: Optional[str] = None, player_number: Optional[int] = None, team_id: Optional[int] = None) -> Player:
        """
        Create a new player.

        Args:
            name: Optional name for the player
            image_path: Optional path to the player's image
            player_number: Optional jersey number of the player (non-negative integer)

        Returns:
            The newly created Player instance
        """
        with self._lock:
            # Basic validation for player_number
            if player_number is not None and player_number < 0:
                raise ValueError("player_number must be a non-negative integer")

            player = self._create_player(name=name, image_path=image_path, player_number=player_number, team_id=team_id)
            return player        

    def _create_player(self, name: Optional[str] = None, image_path: Optional[str] = None, player_number: Optional[int] = None, team_id: Optional[int] = None) -> Player:
        """
        Internal method to create a new player. Does not acquire lock.

        Args:
            name: Optional name for the player
            image_path: Optional path to the player's image
            player_number: Optional jersey number of the player (non-negative integer)

        Returns:
            The newly created Player instance
        """
     
        player_id = self._next_player_id
        self._next_player_id += 1

        player = Player(player_id=player_id, name=name, image_path=image_path, player_number=player_number, team_id=team_id)
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

    def update_player(self, player_id: int, **kwargs) -> Optional[Player]:
        """
        Update a player's attributes.

        Args:
            player_id: ID of the player to update
            **kwargs: Attributes to update (name, track_ids, image_path)

        Returns:
            Updated Player instance if successful, None if player not found or invalid attributes
        """
        with self._lock:
            if player_id not in self.players:
                logger.warning(f"Player {player_id} not found for update")
                return None
            
            # track_ids should not be updated directly
            if "track_ids" in kwargs:
                logger.warning(f"Direct update of 'track_ids' is not allowed for player {player_id}. Use add_track_to_player/remove_track_from_player methods.")
                return None

            # Validate player_number if provided
            if "player_number" in kwargs:
                value = kwargs["player_number"]
                if value is not None and (not isinstance(value, int) or value < 0):
                    logger.warning(f"Invalid player_number '{value}' for player {player_id}. Must be non-negative integer or None.")
                    return None

            # Validate team_id if provided
            if "team_id" in kwargs:
                value = kwargs["team_id"]
                if value is not None and (not isinstance(value, int) or value < 0):
                    logger.warning(f"Invalid team_id '{value}' for player {player_id}. Must be non-negative integer or None.")
                    return None

            # Check for invalid attributes
            invalid_keys = set(kwargs.keys()) - Player.UPDATABLE_ATTRIBUTES
            if invalid_keys:
                logger.warning(f"Attempted to update invalid attributes {invalid_keys} for player {player_id}. Allowed attributes: {Player.UPDATABLE_ATTRIBUTES}")
                return None
            
            # Perform updates if all keys are valid
            for key, value in kwargs.items():
                setattr(self.players[player_id], key, value)
                logger.info(f"Updated {key} for player {player_id} to '{value}'")
            
            return self.players[player_id]

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
           return self._add_track_to_player(player_id, track_id)


    def _add_track_to_player(self, player_id: int, track_id: int) -> bool:
        """
        Internal method to associate a track ID with a player, without acquiring lock.

        Args:
            player_id: ID of the player
            track_id: Track ID to associate

        Returns:
            True if added, False if player not found or track already associated
        """
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
        track_pics = set()


        # Sets initial player image path if not already set
        if player.image_path is None:
            track_path = PlayerManager.path_manager.get_path("unverified_tracks", video_id=self.video_id, track_id=track_id)
            track_pics = self.storage.list_blobs(track_path)
            if not track_pics or len(track_pics) == 0:
                logger.warning(f"No track pictures found for track {track_id} in video {self.video_id}")
            else:
                player.image_path = track_pics.pop() if track_pics else None

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

    def serialize(self) -> str:
        """
        Save player data to JSON file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert track_to_player dict keys/values to native Python ints for JSON serialization
            track_to_player_serializable = {
                int(track_id): int(player_id) 
                for track_id, player_id in self.track_to_player.items()
            }
            
            data = {
                "video_id": self.video_id,
                "next_player_id": int(self._next_player_id),
                "players": [player.to_dict() for player in self.players.values()],
                "track_to_player": track_to_player_serializable
            }
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Failed to serialize player data for video {self.video_id}: {e}")
            return ""

        
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


def initialize_player_manager(video_id: str, current_frame_id: int, frame_detections: Detections, storage) -> Optional[PlayerManager]:
    """
    Initialize a PlayerManager for a given video ID, loading existing data if available.

    Args:
        video_id: Unique identifier for the video
        current_frame_id: The current frame ID being processed
        frame_detections: Detections for the current frame

    Returns:
        Initialized PlayerManager instance if there are detections.tracker_id in frame, None otherwise
    """
    try:
        if frame_detections.tracker_id is None or len(frame_detections.tracker_id) == 0:
            logger.info(f"No tracker IDs in detections for video {video_id} at frame {current_frame_id}. Skipping PlayerManager initialization.")
            return None
        else:
            manager = PlayerManager(video_id=video_id, storage=storage)
            manager.initialize(frame_detections)
        return manager
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PlayerManager for video {video_id}: {e}")
    

def load_player_manager(video_id: str, players_json: str, storage) -> Optional[PlayerManager]:
    """
    Load a PlayerManager from a JSON object.

    Args:
        video_id: Unique identifier for the video
        players_json: JSON object containing player data
        storage: Google Storage client instance

    Returns:
        Loaded PlayerManager instance
    """
    try:
        manager = PlayerManager(video_id, storage=storage)
        manager.load_players_from_json(players_json)
        return manager
    
    except Exception as e:
        logger.error (f"Failed to load PlayerManager for video {video_id}: {e}")
        return None
