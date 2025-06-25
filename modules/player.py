from typing import Optional, Literal, Dict, List, Tuple
import logging
import numpy as np
import supervision as sv
from sklearn.metrics.pairwise import cosine_similarity # For robust cosine similarity calculation

logger = logging.getLogger(__name__)

_REID_SIMILARITY_THRESHOLD = 0.9

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
    """
    _next_id: int = 1
    _registry: Dict[int, 'Player'] = {} # Registry to map tracker IDs to Player objects.

    def __init__(self, initial_embedding: Optional[np.ndarray] = None, initial_crops: Optional[List[np.ndarray]] = None) -> None:

        self.id = Player._next_id
        Player._next_id += 1 
        self.name: Optional[str] = None
        self.jersey_color: Optional[Literal["light", "dark"]] = None
        self.jersey_number: Optional[int] = None
        self.embeddings: Optional[np.ndarray] = initial_embedding
        self.crops: List[np.ndarray] = initial_crops if initial_crops is not None else []
        self.associated_tracker_ids: List[int] = []

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
    def update_or_create(cls, tracker_id: int) ->  'Player':
        """
        Updates an existing player's confirmation count or creates a new player
        if one with the given tracker_id does not exist.

        Args:
            tracker_id: The tracker ID of the player to update or create.

        Returns:
            The player object.
        """
        # This method is for simple presence tracking and does not handle re-ID data.
        # For re-ID, use `match_and_update_for_batch` or `register_new_players_batch`.

        if tracker_id in cls._registry:
            player = cls._registry[tracker_id]

        else:
            player = cls() # Calls __init__
            cls._registry[tracker_id] = player

        return player
    
    @classmethod
    def register_new_players_batch(cls, tracker_ids: List[int], embeddings: List[np.ndarray], crops: List[List[np.ndarray]]) -> List['Player']:
        """
        Registers a batch of truly new players (not found in re-identification).
        Each input tracker_id must not already be in the registry.

        Args:
            tracker_ids (List[int]): List of new tracker IDs.
            embeddings (List[np.ndarray]): List of initial embeddings for each new tracker.
            crops (List[List[np.ndarray]]): List of initial crops for each new tracker.

        Returns:
            List['Player']: A list of newly created Player objects.
        """
        players: List['Player'] = []
        if not (len(tracker_ids) == len(embeddings) == len(crops)):
            raise ValueError("tracker_ids, embeddings, and crops must have the same length.")

        for i in range(len(tracker_ids)):
            tid = tracker_ids[i]
            embedding = embeddings[i]
            crop_list = crops[i]

            if tid in cls._registry:
                logger.warning(f"Attempted to register new player with existing tracker ID {tid}. Skipping.")
                continue

            if embedding is None:
                logger.warning(f"Skipping player creation for tracker ID {tid} due to missing embedding.")
                continue

            player = cls(initial_embedding=embedding, initial_crops=crop_list)
            player.associated_tracker_ids.append(tid)
            cls._registry[tid] = player
            players.append(player)
            logger.debug(f"Registered new player {player.id} with tracker ID {tid}.")
        return players

    @classmethod
    def match_and_update_for_batch(
        cls,
        new_tracker_ids: List[int],
        new_embeddings: List[np.ndarray],
        new_crops: List[List[np.ndarray]],
        orphan_track_ids: List[int]
    ) -> Tuple[List[int], List[np.ndarray], List[List[np.ndarray]], List[int]]:
        """
        Attempts to re-identify new tracker IDs with existing players in the registry
        using cosine similarity.

        If a new tracker ID's embedding is sufficiently similar to an existing player's
        embedding, the new tracker ID is associated with that existing player.
        Otherwise, the new tracker ID, its embedding, and crops are returned as
        unmatched.

        Args:
            new_tracker_ids (List[int]): List of tracker IDs that need to be matched.
            new_embeddings (List[np.ndarray]): List of embeddings corresponding to `new_tracker_ids`.
            new_crops (List[List[np.ndarray]]): List of crops corresponding to `new_tracker_ids`.

        Returns:
            A tuple containing:
            - unmatched_tids: Tracker IDs that could not be matched.
            - unmatched_embeddings: Corresponding embeddings for unmatched TIDs.
            - unmatched_crops: Corresponding crops for unmatched TIDs.
            - reassigned_tids: Tracker IDs that were successfully re-identified.
        """
        if not (len(new_tracker_ids) == len(new_embeddings) == len(new_crops)):
            raise ValueError("new_tracker_ids, new_embeddings, and new_crops must have the same length.")

        unmatched_tids: List[int] = []
        unmatched_embeddings: List[np.ndarray] = []
        unmatched_crops: List[List[np.ndarray]] = []
        reassigned_tids: List[int] = [] # This will store the newly matched TIDs

        # 1. Gather existing player embeddings and their associated Player objects for orphans
        orphan_players_with_embeddings: List[Tuple['Player', np.ndarray]] = []
        
        for tid in orphan_track_ids:
            if tid not in cls._registry:
                logger.warning(f"Orphan track ID {tid} not found in registry. Skipping re-identification for it.")
                continue
            player_obj = cls._registry[tid]
            if player_obj.embeddings is not None:
                orphan_players_with_embeddings.append((player_obj, player_obj.embeddings))

        if not new_tracker_ids or not orphan_players_with_embeddings:
            logger.debug("No new tracks to match or no orphan players to match against. All new tracks considered unmatched.")
            return new_tracker_ids, new_embeddings, new_crops, []

        # Extract embeddings for batch processing
        orphan_embeddings_array = np.array([emb for _, emb in orphan_players_with_embeddings])
        new_embeddings_array = np.array(new_embeddings)


        # 2. Compute Cosine Similarity Matrix
        try:
            similarity_matrix = cosine_similarity(new_embeddings_array, orphan_embeddings_array)
        except ValueError as e:
            logger.error(f"Error computing cosine similarity: {e}. Check embedding dimensions or content.")
            # If similarity computation fails, treat all as unmatched
            return new_tracker_ids, new_embeddings, new_crops, []

        # 3. Match New Tracks to Existing Players
        matched_new_tids_set = set()
        for i, new_tid in enumerate(new_tracker_ids):
            # If this new_tid is already in the registry, it means it was already linked
            # (e.g., by update_or_create in a previous frame). We should not re-identify it.
            if new_tid in cls._registry:
                logger.debug(f"New tracker ID {new_tid} is already in registry. Skipping re-identification for it.")
                matched_new_tids_set.add(new_tid) # Mark as "handled"
                continue

            # Find the best match for the current new_embedding
            best_match_idx = np.argmax(similarity_matrix[i])
            max_similarity = similarity_matrix[i, best_match_idx]

            if max_similarity >= _REID_SIMILARITY_THRESHOLD:
                matched_player, _ = orphan_players_with_embeddings[best_match_idx]
                cls._registry[new_tid] = matched_player
                matched_player.associated_tracker_ids.append(new_tid)
                matched_player.crops.extend(new_crops[i])
                matched_new_tids_set.add(new_tid)
                reassigned_tids.append(new_tid)
                logger.debug(f"Re-identified tracker ID {new_tid} as existing player {matched_player.id} (similarity: {max_similarity:.2f}).")
            else:
                logger.debug(f"Tracker ID {new_tid} (similarity: {max_similarity:.2f}) did not find a strong re-ID match.")
        
        # 4. Collect Unmatched
        for i, original_new_tid in enumerate(new_tracker_ids):
            if original_new_tid not in matched_new_tids_set:
                unmatched_tids.append(original_new_tid)
                unmatched_embeddings.append(new_embeddings[i])
                unmatched_crops.append(new_crops[i])

        return unmatched_tids, unmatched_embeddings, unmatched_crops, reassigned_tids
    
    @classmethod
    def get_registered_tracker_ids(cls) -> List[int]:
        """
        Returns a list of all registered tracker IDs.

        Returns:
            A list of integers representing the tracker IDs of all registered players.
        """
        return list(cls._registry.keys()) # This returns all TIDs that are keys in the registry

    @classmethod
    def get_registered_players(cls) -> List['Player']:
        """
        Returns a list of all registered Player instances.

        Returns:
            A list of Player instances.
        """
        return list(set(cls._registry.values())) # Use set to get unique Player objects
    
    
    
