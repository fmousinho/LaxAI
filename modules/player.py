from typing import Optional, Literal, Dict, List, Tuple
import logging
import numpy as np
import supervision as sv
from sklearn.metrics.pairwise import cosine_similarity # For robust cosine similarity calculation
from scipy.optimize import linear_sum_assignment # For optimal one-to-one matching

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

    
    @classmethod
    def match_and_update_for_batch(
        cls,
        new_tracker_ids: List[int],
        new_embeddings: List[np.ndarray],
        new_crops: List[List[np.ndarray]],
        new_tid_teams: List[int],
        tracker_ids_ineligible_for_match: Optional[List[int]] = None
    ) -> Tuple[List[int], List[np.ndarray], List[List[np.ndarray]], List[int]]:
        """
        Attempts to re-identify new tracker IDs with existing orphan players using
        an optimal one-to-one assignment based on cosine similarity, but only within the same team.

        Args:
            new_tracker_ids (List[int]): List of tracker IDs that need to be matched.
            new_embeddings (List[np.ndarray]): List of embeddings corresponding to `new_tracker_ids`.
            new_crops (List[List[np.ndarray]]): List of crops corresponding to `new_tracker_ids`.
            new_tid_teams (List[int]): List of team IDs corresponding to `new_tracker_ids`.
            tracker_ids_ineligible_for_match (Optional[List[int]]): List of tracker IDs that should not be matched.

        Returns:
            A tuple containing:
            - unmatched_tids: Tracker IDs that could not be matched. (List[int])
            - unmatched_embeddings: Corresponding embeddings for unmatched TIDs. (List[np.ndarray])
            - unmatched_crops: Corresponding crops for unmatched TIDs. (List[List[np.ndarray]])
            - unmatched_tid_teams: Corresponding team IDs for unmatched TIDs. (List[int])
        """
        if not (len(new_tracker_ids) == len(new_embeddings) == len(new_crops) == len(new_tid_teams)):
            raise ValueError("new_tracker_ids, new_embeddings, new_crops, and new_tid_teams must have the same length.")

        unmatched_tids: List[int] = []
        unmatched_embeddings: List[np.ndarray] = []
        unmatched_crops: List[List[np.ndarray]] = []
        unmatched_tid_teams: List[int] = []

        # For each team, match only within that team
        unique_teams = set(new_tid_teams)
        for team in unique_teams:
            # Indices of new_tracker_ids for this team
            team_indices = [i for i, t in enumerate(new_tid_teams) if t == team]
            team_new_tracker_ids = [new_tracker_ids[i] for i in team_indices]
            team_new_embeddings = [new_embeddings[i] for i in team_indices]
            team_new_crops = [new_crops[i] for i in team_indices]
            team_new_tid_teams = [new_tid_teams[i] for i in team_indices]

            # Find eligible players in this team
            players_eligible_for_match = [
                player for player in cls._registry.values()
                if player.team == team and (
                    tracker_ids_ineligible_for_match is None or
                    not set(player.associated_tracker_ids).intersection(tracker_ids_ineligible_for_match)
                )
            ]

            eligible_embeddings_array = np.array([player.embeddings for player in players_eligible_for_match if player.embeddings is not None])
            new_embeddings_array = np.array(team_new_embeddings)

            if eligible_embeddings_array.size == 0 or new_embeddings_array.size == 0:
                # No eligible players or no new embeddings for this team
                unmatched_tids.extend(team_new_tracker_ids)
                unmatched_embeddings.extend(team_new_embeddings)
                unmatched_crops.extend(team_new_crops)
                unmatched_tid_teams.extend(team_new_tid_teams)
                continue

            try:
                similarity_matrix = cosine_similarity(new_embeddings_array, eligible_embeddings_array)
            except ValueError as e:
                logger.error(f"Error computing cosine similarity for team {team}: {e}. Check embedding dimensions or content.")
                unmatched_tids.extend(team_new_tracker_ids)
                unmatched_embeddings.extend(team_new_embeddings)
                unmatched_crops.extend(team_new_crops)
                unmatched_tid_teams.extend(team_new_tid_teams)
                continue

            cost_matrix = 1 - similarity_matrix
            new_track_indices, eligible_player_indices = linear_sum_assignment(cost_matrix)

            matched_new_tids_set = set()
            for i, j in zip(new_track_indices, eligible_player_indices):
                similarity = similarity_matrix[i, j]
                if similarity < _REID_SIMILARITY_THRESHOLD:
                    continue

                new_tid = team_new_tracker_ids[i]
                if new_tid in cls._registry:
                    logger.debug(f"New tracker ID {new_tid} is already in registry. Skipping re-identification for it.")
                    matched_new_tids_set.add(new_tid)
                    continue

                matched_player = players_eligible_for_match[j]
                cls._registry[new_tid] = matched_player
                matched_player.associated_tracker_ids.append(new_tid)
                matched_player.crops.extend(team_new_crops[i])
                matched_player.team = team  # Ensure team is set
                matched_new_tids_set.add(new_tid)
                logger.info(f"Re-identified tracker ID {new_tid} as existing player {matched_player.id} (team {team}, similarity: {similarity:.2f}) via optimal assignment.")

            for idx, original_new_tid in enumerate(team_new_tracker_ids):
                if original_new_tid not in matched_new_tids_set:
                    unmatched_tids.append(original_new_tid)
                    unmatched_embeddings.append(team_new_embeddings[idx])
                    unmatched_crops.append(team_new_crops[idx])
                    unmatched_tid_teams.append(team_new_tid_teams[idx])

        return unmatched_tids, unmatched_embeddings, unmatched_crops, unmatched_tid_teams
