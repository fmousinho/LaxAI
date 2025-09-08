"""
Track to Player Mapping Module

This module provides functionality to map tracking IDs from detections to Player objects,
ensuring that each detection is associated with a Player instance in the registry.

Key Features:
    - Maps track_ids to existing or new Player objects
    - Adds player_id to detection data for downstream processing
    - Handles both existing and new players efficiently
    - Maintains Player registry consistency

Example:
    ```python
    from common.track_to_player import map_detections_to_players

    # detections is sv.Detections with tracker_id
    detections_with_players = map_detections_to_players(detections)
    print(detections_with_players.data['player_id'])  # Array of player IDs
    ```
"""

import logging
import numpy as np
from typing import Optional

import supervision as sv
from common.player import Player

logger = logging.getLogger(__name__)


def map_detections_to_players(detections: sv.Detections) -> sv.Detections:
    """
    Map tracking IDs in detections to Player objects and add player_id to detection data.

    This function processes a set of detections and ensures each track_id is associated
    with a Player object. For existing players in the registry, it uses their existing ID.
    For new track_ids, it creates new Player objects and uses their IDs.

    Args:
        detections: Supervision Detections object containing tracker_id information

    Returns:
        sv.Detections: The same detections object with 'player_id' added to the data field

    Raises:
        ValueError: If detections is None or doesn't contain tracker_id information

    Example:
        ```python
        detections = sv.Detections(
            xyxy=np.array([[10, 10, 20, 20]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            tracker_id=np.array([5])
        )

        detections_with_players = map_detections_to_players(detections)
        player_ids = detections_with_players.data['player_id']
        ```
    """
    if detections is None:
        raise ValueError("Detections cannot be None")

    if detections.is_empty():
        logger.debug("Received empty detections, returning as-is")
        return detections

    # Extract track_ids efficiently
    if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
        track_ids = detections.tracker_id
    else:
        logger.warning("Detections object does not have tracker_id attribute")
        return detections

    # Initialize player_ids array
    player_ids = np.zeros(len(track_ids), dtype=int)

    # Process each track_id
    for i, track_id in enumerate(track_ids):
        # Check if player already exists in registry
        existing_player = Player.get_player_by_tid(track_id)

        if existing_player is not None:
            # Use existing player's ID
            player_ids[i] = existing_player.id
            logger.debug(f"Using existing player {existing_player.id} for track_id {track_id}")
        else:
            # Create new player
            new_player = Player(tracker_id=track_id)
            player_ids[i] = new_player.id
            logger.debug(f"Created new player {new_player.id} for track_id {track_id}")

    # Add player_ids to detections data
    if detections.data is None:
        detections.data = {}

    detections.data['player_id'] = player_ids

    logger.info(f"Mapped {len(track_ids)} detections to {len(set(player_ids))} unique players")

    return detections


def get_player_ids_from_detections(detections: sv.Detections) -> Optional[np.ndarray]:
    """
    Extract player_ids from detections that have been processed by map_detections_to_players.

    Args:
        detections: Supervision Detections object that should contain 'player_id' in data

    Returns:
        np.ndarray or None: Array of player IDs if available, None otherwise
    """
    if detections is None or detections.data is None:
        return None

    return detections.data.get('player_id')


def get_unique_players_from_detections(detections: sv.Detections) -> set:
    """
    Get a set of unique Player objects from detections.

    Args:
        detections: Supervision Detections object with player_id in data

    Returns:
        set: Set of unique Player objects
    """
    player_ids = get_player_ids_from_detections(detections)
    if player_ids is None:
        return set()

    unique_players = set()
    for player_id in np.unique(player_ids):
        player = Player.get_player_by_id(player_id)
        if player is not None:
            unique_players.add(player)

    return unique_players
