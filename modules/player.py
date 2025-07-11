import logging
from typing import Dict, List, Literal, Optional

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from config.transforms_config import player_config

logger = logging.getLogger(__name__)


class Player:
    """
    Represents a player detected and tracked in the video.

    This class manages player identities across multiple detections and frames,
    maintaining embeddings, crops, and metadata for player re-identification
    and team assignment tasks.

    Attributes:
        id: A unique identifier for the player
        tracker_id: The ID assigned by the object tracker (e.g., ByteTrack)
        name: The name of the player, if identified
        jersey_color: The player's jersey color
        jersey_number: The player's jersey number
        crops: List of cropped images of the player
        embeddings: The embedding vector representing the player
        det_class: The detection class ID of the player (e.g., 3 for 'player')
        team: The assigned team ID for the player
    """
    
    _next_id: int = 1
    _registry: Dict[int, 'Player'] = {}  # Registry to map tracker IDs to Player objects

    def __init__(
        self,
        tracker_id: Optional[int] = None,
        initial_embedding: Optional[np.ndarray] = None, 
        initial_crops: Optional[List[np.ndarray]] = None, 
        team: Optional[int] = None
    ) -> None:
        """
        Initialize a new Player instance.
        
        Args:
            tracker_id: Optional tracker ID to associate with this player
            initial_embedding: Optional initial embedding vector
            initial_crops: Optional list of initial crop images
            team: Optional team ID for this player
        """
        self.id = Player._next_id
        Player._next_id += 1 
        self.name: Optional[str] = None
        self.jersey_color: Optional[Literal["light", "dark"]] = None
        self.jersey_number: Optional[int] = None
        self.embeddings: Optional[np.ndarray] = initial_embedding
        self.crops: List[np.ndarray] = initial_crops if initial_crops is not None else []
        self.associated_tracker_ids: List[int] = [tracker_id] if tracker_id is not None else []
        self.team = team
        self.frame_first_seen: int = -1
        self.frame_last_seen: int = -1

    @classmethod
    def get_player_by_tid(cls, tracker_id: int) -> Optional['Player']:
        """
        Retrieve a Player instance from the registry by its tracker_id.

        Args:
            tracker_id: The tracker ID of the player to retrieve

        Returns:
            The Player instance if found, otherwise None
        """
        return cls._registry.get(tracker_id)
    
    @classmethod
    def get_player_by_id(cls, player_id: int) -> Optional['Player']:
        """
        Retrieve a Player instance from the registry by its unique player ID.

        Args:
            player_id: The unique ID of the player to retrieve

        Returns:
            The Player instance if found, otherwise None
        """
        # Iterate through the values (Player objects) in the registry
        # as _registry is keyed by tracker_id, not player.id
        for player in cls._registry.values():
            if player.id == player_id:
                return player
        return None

    def update_embeddings(
        self, 
        new_embedding: np.ndarray, 
        new_det_class: Optional[int] = None
    ) -> None:
        """
        Update the player's embedding and optionally their detection class.

        Args:
            new_embedding: The new embedding to update
            new_det_class: Optional new detection class ID
        """
        self.embeddings = new_embedding
        if new_det_class is not None:
            self.det_class = new_det_class

    def add_crop(self, crop: np.ndarray) -> None:
        """
        Add a new crop image to the player's collection.
        
        Args:
            crop: The crop image to add
        """
        if crop is not None and crop.size > 0:
            self.crops.append(crop)
        else:
            logger.warning(f"Attempted to add invalid crop to player {self.id}")

    def get_similarity_to_embedding(self, other_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity between this player's embedding and another embedding.
        
        Args:
            other_embedding: The embedding to compare against
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if self.embeddings is None or other_embedding is None:
            return 0.0
        
        # Reshape embeddings for sklearn cosine_similarity
        emb1 = self.embeddings.reshape(1, -1)
        emb2 = other_embedding.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)

    def __repr__(self) -> str:
        """Return a string representation of the Player."""
        return f"Player(id={self.id}, team={self.team}, tracker_ids={self.associated_tracker_ids})"

