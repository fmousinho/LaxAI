"""
Player-Track Association Module

This module implements algorithms for associating tracks to players based on embedding 
cosine similarity. It provides both greedy and global optimization approaches to ensure
each track is assigned to exactly one player with optimal similarity scores.

Key Features:
- Greedy chronological association algorithm
- Global optimization using Hungarian algorithm
- Cosine similarity calculation for embeddings
- Temporal constraint enforcement (no overlapping tracks per player)
- Team-based association constraints
"""

import logging
from typing import Optional, List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from modules.tracker import TrackData
from modules.player import Player

logger = logging.getLogger(__name__)


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return float(similarity)


def associate_tracks_to_players_greedy(
    tracks_data: Dict[int, TrackData], 
    similarity_threshold: float
) -> Tuple[set[Player], Dict[int, Player]]:
    """
    Associate tracks to players using a greedy chronological approach.
    
    This algorithm processes tracks in chronological order and assigns each track
    to the best matching existing player or creates a new player if no suitable
    match is found.
    
    Args:
        tracks_data: Dictionary mapping track IDs to track data
        similarity_threshold: Minimum cosine similarity for association
        
    Returns:
        Tuple of (set of unique players, mapping from track ID to Player)
    """
    players: set[Player] = set()
    track_to_player: Dict[int, Player] = {}
    
    # Sort tracks by first frame to process them chronologically
    sorted_tracks = sorted(tracks_data.items(), key=lambda x: x[1].frame_first_seen)
    
    for track_id, track_data in tqdm(sorted_tracks, desc="Creating Players (Greedy)", total=len(sorted_tracks)):
        track_first_frame = track_data.frame_first_seen
        track_last_frame = track_data.frame_last_seen
        track_embedding = track_data.embedding
        track_team = track_data.team
        
        # Find potential player matches (players whose last frame is before this track's first frame)
        potential_players = []
        for player in players:
            if player.frame_last_seen < track_first_frame and player.team == track_team:
                potential_players.append(player)
        
        if len(potential_players) == 0:
            # No potential matches - create new player
            player = Player(
                tracker_id=track_id,
                initial_embedding=track_embedding,
                initial_crops=track_data.crops,
                team=track_team
            )
            # Set the frame bounds for the new player
            player.frame_first_seen = track_first_frame
            player.frame_last_seen = track_last_frame
            players.add(player)
            track_to_player[track_id] = player
            logger.debug(f"Created new player for track {track_id} (no potential matches)")
        else:
            # Calculate cosine similarities with all potential players
            best_match = None
            best_similarity = -1.0
            
            for player in potential_players:
                # Calculate similarity with player's embedding
                if player.embeddings is not None:
                    similarity = cosine_similarity(track_embedding, player.embeddings)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = player
            
            # Associate track with player if similarity is above threshold
            if best_similarity >= similarity_threshold and best_match is not None:
                # Add this track's data to the existing player
                best_match.associated_tracker_ids.append(track_id)
                best_match.crops.extend(track_data.crops)
                best_match.frame_last_seen = track_last_frame
                # Update player's embedding (average with existing)
                if best_match.embeddings is not None:
                    # Average the embeddings
                    best_match.embeddings = (best_match.embeddings + track_embedding) / 2.0
                else:
                    best_match.embeddings = track_embedding
                    
                track_to_player[track_id] = best_match
                logger.debug(f"Associated track {track_id} with existing player {best_match.id} "
                           f"(similarity: {best_similarity:.3f})")
            else:
                # Similarity too low - create new player
                player = Player(
                    tracker_id=track_id,
                    initial_embedding=track_embedding,
                    initial_crops=track_data.crops,
                    team=track_team
                )
                # Set the frame bounds for the new player
                player.frame_first_seen = track_first_frame
                player.frame_last_seen = track_last_frame
                players.add(player)
                track_to_player[track_id] = player
                logger.debug(f"Created new player for track {track_id} "
                           f"(best similarity: {best_similarity:.3f} < {similarity_threshold})")
    
    return players, track_to_player


def associate_tracks_to_players_globally(
    tracks_data: Dict[int, TrackData], 
    similarity_threshold: float
) -> Tuple[set[Player], Dict[int, Player]]:
    """
    Associate tracks to players using global optimization to find the best overall assignment.
    
    This approach processes tracks in batches and uses the Hungarian algorithm to ensure
    optimal similarity assignments within each batch, providing better overall results
    than the greedy approach.
    
    Args:
        tracks_data: Dictionary mapping track IDs to track data
        similarity_threshold: Minimum cosine similarity for association
        
    Returns:
        Tuple of (set of unique players, mapping from track ID to Player)
    """
    
    from scipy.optimize import linear_sum_assignment
    has_scipy = True
    
    players: set[Player] = set()
    track_to_player: Dict[int, Player] = {}
    
    # Sort tracks by first frame to process them chronologically
    sorted_tracks = sorted(tracks_data.items(), key=lambda x: x[1].frame_first_seen)
    
    # Process tracks in batches for global optimization
    BATCH_SIZE = 10  # Process tracks in batches for better optimization
    
    for batch_start in range(0, len(sorted_tracks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(sorted_tracks))
        batch_tracks = sorted_tracks[batch_start:batch_end]
        
        if batch_tracks and has_scipy:
            # Find potential players for all tracks in this batch
            all_potential_players = set()
            track_player_candidates = {}
            
            for track_id, track_data in batch_tracks:
                track_first_frame = track_data.frame_first_seen
                track_team = track_data.team
                
                # Find potential player matches
                potential_players = []
                for player in players:
                    if player.frame_last_seen < track_first_frame and player.team == track_team:
                        potential_players.append(player)
                
                track_player_candidates[track_id] = potential_players
                all_potential_players.update(potential_players)
            
            if all_potential_players:
                # Use Hungarian algorithm for optimal assignment within this batch
                batch_assignments = find_optimal_batch_assignment(
                    batch_tracks, list(all_potential_players), similarity_threshold
                )
                
                # Sort assignments by track frame order to prevent overlap issues
                sorted_assignments = []
                for track_id, track_data in batch_tracks:
                    assigned_player = batch_assignments.get(track_id)
                    if assigned_player is not None:
                        sorted_assignments.append((track_id, track_data, assigned_player))
                
                # Sort by frame_first_seen to process tracks chronologically
                sorted_assignments.sort(key=lambda x: x[1].frame_first_seen)
                
                # Apply assignments in chronological order
                for track_id, track_data, assigned_player in sorted_assignments:
                    # Double-check that this assignment is still valid
                    if assigned_player.frame_last_seen < track_data.frame_first_seen and assigned_player.team == track_data.team:
                        # Associate with existing player
                        assigned_player.associated_tracker_ids.append(track_id)
                        assigned_player.crops.extend(track_data.crops)  #
                        assigned_player.frame_last_seen = track_data.frame_last_seen
                        
                        # Update player's embedding (average with existing)
                        if assigned_player.embeddings is not None:
                            # Average the embeddings
                            assigned_player.embeddings = (assigned_player.embeddings + track_data.embedding) / 2.0
                        else:
                            assigned_player.embeddings = track_data.embedding
                        
                        track_to_player[track_id] = assigned_player
                        logger.debug(f"Globally assigned track {track_id} to player {assigned_player.id}")
                    else:
                        # Assignment is no longer valid, create new player
                        player = Player(
                            tracker_id=track_id,
                            initial_embedding=track_data.embedding,
                            initial_crops=track_data.crops,
                            team=track_data.team
                        )
                        # Set the frame bounds for the new player
                        player.frame_first_seen = track_data.frame_first_seen
                        player.frame_last_seen = track_data.frame_last_seen
                        players.add(player)
                        track_to_player[track_id] = player
                        logger.debug(f"Created new player for track {track_id} (assignment became invalid)")
                
                # Handle unassigned tracks from this batch
                assigned_track_ids = {assignment[0] for assignment in sorted_assignments}
                for track_id, track_data in batch_tracks:
                    if track_id not in assigned_track_ids:
                        # Create new player for unassigned track
                        player = Player(
                            tracker_id=track_id,
                            initial_embedding=track_data.embedding,
                            initial_crops=track_data.crops,
                            team=track_data.team
                        )
                        # Set the frame bounds for the new player
                        player.frame_first_seen = track_data.frame_first_seen
                        player.frame_last_seen = track_data.frame_last_seen
                        players.add(player)
                        track_to_player[track_id] = player
                        logger.debug(f"Created new player for track {track_id} (unassigned in batch)")
            else:
                # No potential players, create new players for all tracks in batch
                for track_id, track_data in batch_tracks:
                    player = Player(
                        tracker_id=track_id,
                        initial_embedding=track_data.embedding,
                        initial_crops=track_data.crops,
                        team=track_data.team
                    )
                    # Set the frame bounds for the new player
                    player.frame_first_seen = track_data.frame_first_seen
                    player.frame_last_seen = track_data.frame_last_seen
                    players.add(player)
                    track_to_player[track_id] = player
                    logger.debug(f"Created new player for track {track_id} (no potential players)")
        else:
            # Fall back to greedy for this batch
            for track_id, track_data in batch_tracks:
                potential_players = track_player_candidates[track_id]
                
                if len(potential_players) == 0:
                    # No potential matches - create new player
                    player = Player(
                        tracker_id=track_id,
                        initial_embedding=track_data.embedding,
                        initial_crops=track_data.crops,
                        team=track_data.team
                    )
                    # Set the frame bounds for the new player
                    player.frame_first_seen = track_data.frame_first_seen
                    player.frame_last_seen = track_data.frame_last_seen
                    players.add(player)
                    track_to_player[track_id] = player
                    logger.debug(f"Created new player for track {track_id} (no potential matches)")
                else:
                    # Find best match using greedy approach
                    best_match = None
                    best_similarity = -1.0
                    
                    for player in potential_players:
                        max_player_similarity = -1.0
                        if player.embeddings is not None:
                            similarity = cosine_similarity(track_data.embedding, player.embeddings)
                            max_player_similarity = similarity
                        
                        if max_player_similarity > best_similarity:
                            best_similarity = max_player_similarity
                            best_match = player
                    
                    # Associate if above threshold
                    if best_similarity >= similarity_threshold and best_match is not None:
                        best_match.associated_tracker_ids.append(track_id)
                        best_match.crops.extend(track_data.crops)
                        best_match.frame_last_seen = track_data.frame_last_seen
                        
                        # Update player's embedding (average with existing)
                        if best_match.embeddings is not None:
                            # Average the embeddings
                            best_match.embeddings = (best_match.embeddings + track_data.embedding) / 2.0
                        else:
                            best_match.embeddings = track_data.embedding
                        
                        track_to_player[track_id] = best_match
                        logger.debug(f"Associated track {track_id} with player {best_match.id} "
                                   f"(similarity: {best_similarity:.3f})")
                    else:
                        # Create new player
                        player = Player(
                            tracker_id=track_id,
                            initial_embedding=track_data.embedding,
                            initial_crops=track_data.crops,
                            team=track_data.team
                        )
                        # Set the frame bounds for the new player
                        player.frame_first_seen = track_data.frame_first_seen
                        player.frame_last_seen = track_data.frame_last_seen
                        players.add(player)
                        track_to_player[track_id] = player
                        logger.debug(f"Created new player for track {track_id} "
                                   f"(best similarity: {best_similarity:.3f} < {similarity_threshold})")
    
    return players, track_to_player


def find_optimal_batch_assignment(
    batch_tracks: List[Tuple[int, TrackData]],
    potential_players: List[Player],
    similarity_threshold: float
) -> Dict[int, Optional[Player]]:
    """
    Find optimal assignment for a batch of tracks using Hungarian algorithm.
    
    This function creates a cost matrix based on cosine similarities and uses
    the Hungarian algorithm to find the assignment that maximizes the sum of
    similarities across all tracks in the batch.
    
    Args:
        batch_tracks: List of (track_id, track_data) pairs
        potential_players: List of potential player matches
        similarity_threshold: Minimum similarity for valid association
        
    Returns:
        Dictionary mapping track IDs to assigned players (or None)
    """
    if not batch_tracks or not potential_players:
        return {}
    
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return {}
    
    # Build cost matrix
    n_tracks = len(batch_tracks)
    n_players = len(potential_players)
    
    # Create cost matrix (negative similarities as costs to maximize)
    cost_matrix = np.full((n_tracks, n_players), 1.0)  # High cost for invalid pairs
    
    for i, (track_id, track_data) in enumerate(batch_tracks):
        track_embedding = track_data.embedding
        track_team = track_data.team
        track_first_frame = track_data.frame_first_seen
        
        for j, player in enumerate(potential_players):
            # Check if this player is a valid candidate
            if player.frame_last_seen >= track_first_frame or player.team != track_team:
                continue  # Invalid assignment
            
            # Calculate similarity with this player's embedding
            max_similarity = -1.0
            if player.embeddings is not None:
                similarity = cosine_similarity(track_embedding, player.embeddings)
                max_similarity = similarity
            
            if max_similarity >= similarity_threshold:
                cost_matrix[i, j] = 1.0 - max_similarity  # Convert to cost
    
    # Solve assignment problem
    track_indices, player_indices = linear_sum_assignment(cost_matrix)
    
    # Create assignments
    assignments = {}
    for track_idx, player_idx in zip(track_indices, player_indices):
        track_id, _ = batch_tracks[track_idx]
        
        if cost_matrix[track_idx, player_idx] < 1.0:  # Valid assignment
            player = potential_players[player_idx]
            assignments[track_id] = player
            similarity = 1.0 - cost_matrix[track_idx, player_idx]
            logger.debug(f"Optimal assignment: track {track_id} -> player {player.id} "
                        f"(similarity: {similarity:.3f})")
        else:
            assignments[track_id] = None  # No valid assignment
    
    return assignments


def find_global_optimal_associations(
    unassigned_tracks: List[Tuple[int, TrackData]],
    potential_players: List[Player],
    similarity_threshold: float
) -> Dict[int, Optional[Player]]:
    """
    Find globally optimal track-to-player associations using Hungarian algorithm.
    
    This function finds the assignment that maximizes the sum of similarities
    across all unassigned tracks. It's useful for cases where you want to
    optimize the entire assignment at once.
    
    Args:
        unassigned_tracks: List of (track_id, track_data) pairs to assign
        potential_players: List of potential player matches
        similarity_threshold: Minimum similarity for valid association
        
    Returns:
        Dictionary mapping track IDs to Player objects (or None if no valid assignment)
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.warning("scipy not available for global optimization")
        return {}
    
    if not unassigned_tracks or not potential_players:
        return {}
    
    # Build similarity matrix
    n_tracks = len(unassigned_tracks)
    n_players = len(potential_players)
    
    # Create cost matrix (we'll use negative similarities as costs to maximize)
    cost_matrix = np.full((n_tracks, n_players), 1.0)  # High cost for invalid pairs
    
    for i, (track_id, track_data) in enumerate(unassigned_tracks):
        track_embedding = track_data.embedding
        
        for j, player in enumerate(potential_players):
            # Calculate similarity with this player's embedding
            max_similarity = -1.0
            
            if player.embeddings is not None:
                similarity = cosine_similarity(track_embedding, player.embeddings)
                max_similarity = similarity
            
            if max_similarity >= similarity_threshold:
                cost_matrix[i, j] = 1.0 - max_similarity  # Convert to cost (lower is better)
    
    # Solve assignment problem
    track_indices, player_indices = linear_sum_assignment(cost_matrix)
    
    # Create associations
    associations = {}
    for track_idx, player_idx in zip(track_indices, player_indices):
        track_id, _ = unassigned_tracks[track_idx]
        
        if cost_matrix[track_idx, player_idx] < 1.0:  # Valid association
            player = potential_players[player_idx]
            associations[track_id] = player
            logger.debug(f"Globally assigned track {track_id} to player {player.id} "
                        f"(similarity: {1.0 - cost_matrix[track_idx, player_idx]:.3f})")
        else:
            associations[track_id] = None  # No valid association
            logger.debug(f"No valid assignment for track {track_id}")
    
    return associations
