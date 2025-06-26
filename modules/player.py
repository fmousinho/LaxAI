from typing import Optional, Literal, Dict, List, Tuple
import logging
import numpy as np
import supervision as sv
from sklearn.metrics.pairwise import cosine_similarity # For robust cosine similarity calculation
from scipy.optimize import linear_sum_assignment # For optimal one-to-one matching

logger = logging.getLogger(__name__)

_REID_SIMILARITY_THRESHOLD = 0.93

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
    
    def update_embeddings(self, new_embedding: np.ndarray) -> None:
        """
        Updates the player's embedding with a new one.

        Args:
            new_embedding (np.ndarray): The new embedding to update.
        """
        
        self.embeddings = new_embedding

    
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
        tracker_ids_ineligible_for_match: Optional[List[int]] = None
    ) -> Tuple[List[int], List[np.ndarray], List[List[np.ndarray]]]:
        """
        Attempts to re-identify new tracker IDs with existing orphan players using
        an optimal one-to-one assignment based on cosine similarity.

        If a new tracker ID's embedding is sufficiently similar to an existing player's
        embedding, the new tracker ID is associated with that existing player.
        Otherwise, the new tracker ID, its embedding, and crops are returned as
        unmatched.

        Args:
            new_tracker_ids (List[int]): List of tracker IDs that need to be matched.
            new_embeddings (List[np.ndarray]): List of embeddings corresponding to `new_tracker_ids`.
            new_crops (List[List[np.ndarray]]): List of crops corresponding to `new_tracker_ids`.
            tracker_ids_ineligible_for_match (Optional[List[int]]): List of tracker IDs that should not be matched
                                        (because they are already shown in the current frame).

                
        Returns:
            A tuple containing:
            - unmatched_tids: Tracker IDs that could not be matched.
            - unmatched_embeddings: Corresponding embeddings for unmatched TIDs.
            - unmatched_crops: Corresponding crops for unmatched TIDs.
        """
        if not (len(new_tracker_ids) == len(new_embeddings) == len(new_crops)):
            raise ValueError("new_tracker_ids, new_embeddings, and new_crops must have the same length.")

        unmatched_tids: List[int] = []
        unmatched_embeddings: List[np.ndarray] = []
        unmatched_crops: List[List[np.ndarray]] = []

        # Finding players that are eligible for matching
        # We get all players in registry, and remove those in the current frame
        players_eligible_for_match = set(cls._registry.values())
        if tracker_ids_ineligible_for_match is not None:
            for tid in tracker_ids_ineligible_for_match:
                if tid in cls._registry:
                   players_eligible_for_match.discard(cls._registry[tid])
        players_eligible_for_match = list(players_eligible_for_match)

        eligible_embeddings_array = np.array([player.embeddings for player in players_eligible_for_match if player.embeddings is not None])

        new_embeddings_array = np.array(new_embeddings)

        # 2. Compute Cosine Similarity Matrix
        if eligible_embeddings_array.size == 0 or new_embeddings_array.size == 0:
            logger.debug("No eligible players for matching or no new embeddings provided. Returning unmatched lists.")
            return new_tracker_ids, new_embeddings, new_crops
        
        try:
            similarity_matrix = cosine_similarity(new_embeddings_array, eligible_embeddings_array)
        except ValueError as e:
            logger.error(f"Error computing cosine similarity: {e}. Check embedding dimensions or content.")
            # If similarity computation fails, treat all as unmatched
            return new_tracker_ids, new_embeddings, new_crops

        # 3. Find the optimal one-to-one assignment to maximize similarity.
        # The linear_sum_assignment function finds a minimum weight matching.
        # We want to maximize similarity, so we work with cost = 1 - similarity.
        cost_matrix = 1 - similarity_matrix
        
        # Get the indices of the optimal assignments
        new_track_indices, eligible_player_indices = linear_sum_assignment(cost_matrix)

        # 4. Process the optimal matches
        matched_new_tids_set = set()
        for i, j in zip(new_track_indices, eligible_player_indices):
            similarity = similarity_matrix[i, j]

            if similarity < _REID_SIMILARITY_THRESHOLD:
                # Since assignments are sorted by cost, we can stop early
                # if we fall below the similarity threshold.
                continue

            new_tid = new_tracker_ids[i]
            if new_tid in cls._registry:
                logger.debug(f"New tracker ID {new_tid} is already in registry. Skipping re-identification for it.")
                matched_new_tids_set.add(new_tid)
                continue

            matched_player = players_eligible_for_match[j]
            cls._registry[new_tid] = matched_player
            matched_player.associated_tracker_ids.append(new_tid)
            matched_player.crops.extend(new_crops[i])
            matched_new_tids_set.add(new_tid)
            logger.info(f"Re-identified tracker ID {new_tid} as existing player {matched_player.id} (similarity: {similarity:.2f}) via optimal assignment.")

        # 5. Collect Unmatched
        for i, original_new_tid in enumerate(new_tracker_ids):
            if original_new_tid not in matched_new_tids_set:
                unmatched_tids.append(original_new_tid)
                unmatched_embeddings.append(new_embeddings[i])
                unmatched_crops.append(new_crops[i])

        return unmatched_tids, unmatched_embeddings, unmatched_crops
