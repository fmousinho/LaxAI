from typing import Optional, Literal, Dict, List, Tuple
import logging
import numpy as np
import supervision as sv
from sklearn.metrics.pairwise import cosine_similarity # For robust cosine similarity calculation
from scipy.optimize import linear_sum_assignment # For optimal one-to-one matching

logger = logging.getLogger(__name__)

from config.transforms_config import player_config

class Player:
    """
    Represents a player detected and tracked in the video.

    Attributes:
        id (int): A unique identifier for the player.
        tracker_id (int): The ID assigned by the object tracker (e.g., ByteTrack).
        name (Optional[str]): The name of the player, if identified.
        jersey_color (Optional[Literal["light", "dark"]]): The player's jersey color.
        jersey_number (Optional[int]): The player's jersey number.
        crops (List[np.ndarray]): List of cropped images of the player.
        embeddings (Optional[np.ndarray]): The embedding vector representing the player.
        det_class (Optional[int]): The detection class ID of the player (e.g., 3 for 'player').
        team (Optional[int]): The assigned team ID for the player.
    """
    _next_id: int = 1
    _registry: Dict[int, 'Player'] = {} # Registry to map tracker IDs to Player objects.

    def __init__(self,
                 tracker_id: Optional[int] = None,
                 initial_embedding: Optional[np.ndarray] = None, 
                 initial_crops: Optional[List[np.ndarray]] = None, 
                 team: Optional[int] = None
                 ) -> None:

        self.id = Player._next_id
        Player._next_id += 1 
        self.name: Optional[str] = None
        self.jersey_color: Optional[Literal["light", "dark"]] = None
        self.jersey_number: Optional[int] = None
        self.embeddings: Optional[np.ndarray] = initial_embedding
        self.crops: List[np.ndarray] = initial_crops if initial_crops is not None else []
        self.associated_tracker_ids: List[int] = [tracker_id] if tracker_id is not None else []  # List of tracker IDs associated with this player
        self.team = team
        self.frame_first_seen: int = -1
        self.frame_last_seen: int = -1

    @classmethod
    def get_player_by_tid(cls, tracker_id: int) -> Optional['Player']:
        """
        Retrieves a Player instance from the registry by its tracker_id.

        Args:
            tracker_id: The tracker ID of the player to retrieve.

        Returns:
            The Player instance if found, otherwise None.
        """
        return cls._registry.get(tracker_id)
    
    @classmethod
    def get_player_by_id(cls, player_id: int) -> Optional['Player']:
        """
        Retrieves a Player instance from the registry by its unique player ID.

        Args:
            player_id: The unique ID of the player to retrieve.

        Returns:
            The Player instance if found, otherwise None.
        """
        # Iterate through the values (Player objects) in the registry
        # as _registry is keyed by tracker_id, not player.id.
        for player in cls._registry.values():
            if player.id == player_id:
                return player
        return None

    def update_embeddings(self, new_embedding: np.ndarray, new_det_class: Optional[int] = None) -> None:
        """
        Updates the player's embedding and optionally their detection class with new ones.

        Args:
            new_embedding (np.ndarray): The new embedding to update.
        """
        
        self.embeddings = new_embedding
        if new_det_class is not None:
            self.det_class = new_det_class

