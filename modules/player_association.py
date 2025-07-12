"""
Player-Track Association Module

This module implements algorithms for associating tracks to players based on embedding 
cosine similarity. It provides both greedy and global optimization approaches to ensure
each track is assigned to exactly one player with optimal similarity scores.

Key Features:
- Track stitching algorithm to merge fragmented tracklets
- Greedy chronological association algorithm
- Global optimization using Hungarian algorithm
- Cosine similarity calculation for embeddings
- Temporal constraint enforcement (no overlapping tracks per player)
- Team-based association constraints
"""

import logging
from typing import Optional, List, Dict, Tuple
from collections import Counter
import numpy as np

from modules.tracker import TrackData
from modules.player import Player
from modules.utils import l2_normalize_embedding
from config.transforms_config import player_config, track_stitching_config, model_config

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


def calculate_stitching_cost(
    track_a: TrackData,
    track_b: TrackData,
    appearance_weight: float = track_stitching_config.appearance_weight,
    temporal_weight: float = track_stitching_config.temporal_weight,
    motion_weight: float = track_stitching_config.motion_weight,
    max_time_gap: int = track_stitching_config.max_time_gap
) -> float:
    """
    Calculate the cost of stitching two tracklets together.
    
    Args:
        track_a: First tracklet (ending earlier)
        track_b: Second tracklet (starting later)
        appearance_weight: Weight for appearance similarity cost
        temporal_weight: Weight for temporal gap cost
        motion_weight: Weight for motion prediction cost
        max_time_gap: Maximum allowable time gap between tracks
        
    Returns:
        Total stitching cost (lower is better, np.inf if impossible)
    """
    # Check basic constraints
    if track_a.frame_last_seen >= track_b.frame_first_seen:
        return np.inf  # Overlapping tracks cannot be stitched
    
    if track_a.team != track_b.team:
        return np.inf  # Different teams cannot be stitched
    
    time_gap = track_b.frame_first_seen - track_a.frame_last_seen
    if time_gap > max_time_gap:
        return np.inf  # Time gap too large
    
    # Calculate appearance cost (Re-ID similarity)
    appearance_cost = 0.0
    if track_a.embedding.size > 0 and track_b.embedding.size > 0:
        similarity = cosine_similarity(track_a.embedding, track_b.embedding)
        appearance_cost = 1.0 - similarity  # Convert similarity to cost
    else:
        appearance_cost = 1.0  # High cost if no embeddings available
    
    # Calculate temporal cost
    temporal_cost = time_gap * temporal_weight
    
    # Calculate motion cost (simplified for now)
    motion_cost = 0.0
    
    # Combine costs
    total_cost = (appearance_weight * appearance_cost + 
                  temporal_weight * temporal_cost + 
                  motion_weight * motion_cost)
    
    return total_cost


def create_representative_embedding(track_data: TrackData) -> np.ndarray:
    """
    Create a representative embedding for a tracklet.
    
    Args:
        track_data: Track data containing embedding information
        
    Returns:
        Representative embedding vector
    """
    if track_data.embedding.size > 0:
        normalized_embedding = l2_normalize_embedding(track_data.embedding)
        # Ensure we return a numpy array
        if isinstance(normalized_embedding, np.ndarray):
            return normalized_embedding
        else:
            return normalized_embedding.cpu().numpy()
    else:
        # Return zero embedding if no embedding available
        return np.zeros(model_config.embedding_dim)


def update_detection_metadata(frame_detections, track_id_mapping: Dict[int, int]) -> Tuple[int, int]:
    """
    Update detection metadata with new track IDs.
    
    Args:
        frame_detections: Detection object for a frame
        track_id_mapping: Mapping from old to new track IDs
        
    Returns:
        Tuple of (metadata_updates_count, data_updates_count)
    """
    metadata_updates_count = 0
    data_updates_count = 0
    
    # Update any metadata in the data dictionary that references tracker IDs
    if hasattr(frame_detections, 'data') and frame_detections.data:
        for key, values in frame_detections.data.items():
            # Check if this data field contains tracker ID references
            if 'track' in key.lower() and isinstance(values, (list, np.ndarray)):
                # Update tracker ID references in data fields
                updated_values = []
                for value in values:
                    if isinstance(value, (int, np.integer)):
                        updated_values.append(track_id_mapping.get(int(value), int(value)))
                    else:
                        updated_values.append(value)
                frame_detections.data[key] = np.array(updated_values) if isinstance(values, np.ndarray) else updated_values
                data_updates_count += 1
                logger.debug(f"Updated data field '{key}' with stitched track IDs")
            
            # Also check for any integer values that match known track IDs
            elif isinstance(values, (list, np.ndarray)):
                updated_values = []
                field_updated = False
                for value in values:
                    if isinstance(value, (int, np.integer)) and int(value) in track_id_mapping:
                        updated_values.append(track_id_mapping[int(value)])
                        field_updated = True
                    else:
                        updated_values.append(value)
                if field_updated:
                    frame_detections.data[key] = np.array(updated_values) if isinstance(values, np.ndarray) else updated_values
                    data_updates_count += 1
                    logger.debug(f"Updated data field '{key}' with stitched track IDs (general check)")
            
            # Handle single integer values
            elif isinstance(values, (int, np.integer)) and int(values) in track_id_mapping:
                frame_detections.data[key] = track_id_mapping[int(values)]
                data_updates_count += 1
                logger.debug(f"Updated data field '{key}' with stitched track ID (single value)")
    
    # Update any metadata that references tracker IDs
    if hasattr(frame_detections, 'metadata') and frame_detections.metadata:
        for key, value in frame_detections.metadata.items():
            if 'track' in key.lower():
                if isinstance(value, (int, np.integer)):
                    frame_detections.metadata[key] = track_id_mapping.get(int(value), int(value))
                    metadata_updates_count += 1
                    logger.debug(f"Updated metadata field '{key}' with stitched track ID")
                elif isinstance(value, (list, np.ndarray)):
                    updated_values = [track_id_mapping.get(int(v), int(v)) if isinstance(v, (int, np.integer)) else v for v in value]
                    frame_detections.metadata[key] = np.array(updated_values) if isinstance(value, np.ndarray) else updated_values
                    metadata_updates_count += 1
                    logger.debug(f"Updated metadata field '{key}' with stitched track IDs")
            
            # Also check for any integer values that match known track IDs
            elif isinstance(value, (int, np.integer)) and int(value) in track_id_mapping:
                frame_detections.metadata[key] = track_id_mapping[int(value)]
                metadata_updates_count += 1
                logger.debug(f"Updated metadata field '{key}' with stitched track ID (general check)")
            
            elif isinstance(value, (list, np.ndarray)):
                updated_values = []
                field_updated = False
                for v in value:
                    if isinstance(v, (int, np.integer)) and int(v) in track_id_mapping:
                        updated_values.append(track_id_mapping[int(v)])
                        field_updated = True
                    else:
                        updated_values.append(v)
                if field_updated:
                    frame_detections.metadata[key] = np.array(updated_values) if isinstance(value, np.ndarray) else updated_values
                    metadata_updates_count += 1
                    logger.debug(f"Updated metadata field '{key}' with stitched track IDs (general check)")
    
    return metadata_updates_count, data_updates_count


def stitch_tracks(
    tracks_data: Dict[int, TrackData],
    multi_frame_detections: List,
    similarity_threshold: float = track_stitching_config.stitch_similarity_threshold,
    max_time_gap: int = track_stitching_config.max_time_gap,
    appearance_weight: float = track_stitching_config.appearance_weight,
    temporal_weight: float = track_stitching_config.temporal_weight,
    motion_weight: float = track_stitching_config.motion_weight
) -> Tuple[Dict[int, TrackData], List, Dict[int, int]]:
    """
    Stitch fragmented tracklets together to create longer, more complete tracks.
    
    Args:
        tracks_data: Dictionary mapping track IDs to track data
        multi_frame_detections: List of detections for each frame to be updated
        similarity_threshold: Minimum similarity threshold for stitching
        max_time_gap: Maximum frame gap allowed between tracklets
        appearance_weight: Weight for appearance similarity in cost calculation
        temporal_weight: Weight for temporal gap in cost calculation
        motion_weight: Weight for motion prediction in cost calculation
        
    Returns:
        Tuple of (stitched tracks dictionary, updated multi_frame_detections, track_id_mapping)
    """
    logger.info(f"Starting track stitching with {len(tracks_data)} original tracks")
    
    # Create a copy of tracks data to modify
    stitched_tracks = {}
    track_id_mapping = {}  # Maps original track IDs to stitched track IDs
    
    # Sort tracks by start time
    sorted_tracks = sorted(tracks_data.items(), key=lambda x: x[1].frame_first_seen)
    
    # Keep track of which tracks have been stitched
    stitched_track_ids = set()
    next_stitched_id = max(tracks_data.keys()) + 1
    
    # Process each track
    for track_id, track_data in sorted_tracks:
        if track_id in stitched_track_ids:
            continue  # Already stitched
        
        # Create representative embedding for this track
        track_embedding = create_representative_embedding(track_data)
        
        # Find potential tracks to stitch with
        potential_matches = []
        for other_id, other_data in tracks_data.items():
            if (other_id != track_id and 
                other_id not in stitched_track_ids and
                other_data.frame_first_seen > track_data.frame_last_seen):
                
                cost = calculate_stitching_cost(
                    track_data, other_data,
                    appearance_weight, temporal_weight, motion_weight, max_time_gap
                )
                
                if cost < np.inf:
                    potential_matches.append((other_id, other_data, cost))
        
        # Sort by cost (lowest first)
        potential_matches.sort(key=lambda x: x[2])
        
        # Start building the stitched track
        current_stitched_track = TrackData(
            track_id=next_stitched_id,
            crop=track_data.crops[0] if track_data.crops else np.array([]),
            class_id=track_data.class_id,
            confidence=1.0,
            frame_id=track_data.frame_first_seen
        )
        
        # Copy all data from the first track
        current_stitched_track.crops = track_data.crops.copy()
        current_stitched_track.embedding = track_embedding
        current_stitched_track.team = track_data.team
        current_stitched_track.frame_first_seen = track_data.frame_first_seen
        current_stitched_track.frame_last_seen = track_data.frame_last_seen
        
        # Track the original track IDs that were stitched
        stitched_original_ids = [track_id]
        stitched_track_ids.add(track_id)
        
        # Greedily stitch tracks
        current_track_data = track_data
        while potential_matches:
            # Find the best match for the current track
            best_match = None
            best_cost = np.inf
            
            for idx, (other_id, other_data, cost) in enumerate(potential_matches):
                if other_id in stitched_track_ids:
                    continue
                
                # Recalculate cost with current track state
                updated_cost = calculate_stitching_cost(
                    current_track_data, other_data,
                    appearance_weight, temporal_weight, motion_weight, max_time_gap
                )
                
                if updated_cost < best_cost:
                    # Check similarity threshold
                    if (other_data.embedding.size > 0 and 
                        current_track_data.embedding.size > 0):
                        similarity = cosine_similarity(
                            current_track_data.embedding, other_data.embedding
                        )
                        if similarity >= similarity_threshold:
                            best_match = (other_id, other_data, updated_cost)
                            best_cost = updated_cost
            
            if best_match is None:
                break  # No more valid matches
            
            # Stitch the best match
            other_id, other_data, _ = best_match
            stitched_track_ids.add(other_id)
            stitched_original_ids.append(other_id)
            
            # Merge the track data
            current_stitched_track.crops.extend(other_data.crops)
            current_stitched_track.frame_last_seen = other_data.frame_last_seen
            
            # Update embedding (average)
            if other_data.embedding.size > 0:
                averaged_embedding = (current_stitched_track.embedding + other_data.embedding) / 2.0
                normalized_embedding = l2_normalize_embedding(averaged_embedding)
                # Ensure we set a numpy array
                if isinstance(normalized_embedding, np.ndarray):
                    current_stitched_track.embedding = normalized_embedding
                else:
                    current_stitched_track.embedding = normalized_embedding.cpu().numpy()
            
            # Update current track data for next iteration
            current_track_data = other_data
            
            # Remove matched tracks from potential matches
            potential_matches = [
                (oid, odata, cost) for oid, odata, cost in potential_matches
                if oid not in stitched_track_ids
            ]
            
            # Recalculate costs for remaining potential matches
            updated_matches = []
            for oid, odata, _ in potential_matches:
                new_cost = calculate_stitching_cost(
                    current_track_data, odata,
                    appearance_weight, temporal_weight, motion_weight, max_time_gap
                )
                if new_cost < np.inf:
                    updated_matches.append((oid, odata, new_cost))
            
            potential_matches = sorted(updated_matches, key=lambda x: x[2])
        
        # Store the stitched track
        stitched_tracks[next_stitched_id] = current_stitched_track
        
        # Update mapping
        for original_id in stitched_original_ids:
            track_id_mapping[original_id] = next_stitched_id
        
        logger.debug(f"Stitched track {next_stitched_id} from {len(stitched_original_ids)} "
                    f"original tracks: {stitched_original_ids}")
        
        next_stitched_id += 1
    
    # Add any remaining unstitched tracks
    for track_id, track_data in tracks_data.items():
        if track_id not in stitched_track_ids:
            stitched_tracks[track_id] = track_data
            track_id_mapping[track_id] = track_id
    
    logger.info(f"Track stitching complete: {len(tracks_data)} -> {len(stitched_tracks)} tracks")
    logger.info(f"Stitching reduced track count by {len(tracks_data) - len(stitched_tracks)} "
                f"({100 * (len(tracks_data) - len(stitched_tracks)) / len(tracks_data):.1f}%)")
    
    # Update multi_frame_detections with new track IDs
    logger.info("Updating multi_frame_detections with stitched track IDs")
    updated_detections = []
    total_metadata_updates = 0
    total_data_updates = 0
    
    for frame_detections in multi_frame_detections:
        if frame_detections.tracker_id is not None:
            # Update tracker IDs based on the mapping
            new_tracker_ids = []
            for old_id in frame_detections.tracker_id:
                new_id = track_id_mapping.get(old_id, old_id)
                new_tracker_ids.append(new_id)
            
            # Update the tracker_id array
            frame_detections.tracker_id = np.array(new_tracker_ids)
            
            # Update metadata
            metadata_updates, data_updates = update_detection_metadata(frame_detections, track_id_mapping)
            total_metadata_updates += metadata_updates
            total_data_updates += data_updates
            
            updated_detections.append(frame_detections)
        else:
            # No tracker IDs to update
            updated_detections.append(frame_detections)
    
    logger.info(f"Updated {len(updated_detections)} frames with new track IDs")
    if total_metadata_updates > 0:
        logger.info(f"Updated {total_metadata_updates} metadata fields with stitched track IDs")
    if total_data_updates > 0:
        logger.info(f"Updated {total_data_updates} data fields with stitched track IDs")
    
    return stitched_tracks, updated_detections, track_id_mapping


def associate_tracks_to_players_greedy(
    tracks_data: Dict[int, TrackData], 
    similarity_threshold: float = player_config.reid_similarity_threshold
) -> Tuple[set[Player], Dict[int, Player]]:
    """
    Associate tracks to players using a greedy chronological approach.
    
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
    
    for track_id, track_data in sorted_tracks:
        track_first_frame = track_data.frame_first_seen
        track_last_frame = track_data.frame_last_seen
        track_embedding = track_data.embedding
        track_team = track_data.team
        
        # Find potential player matches
        potential_players = [
            player for player in players
            if player.frame_last_seen < track_first_frame and player.team == track_team
        ]
        
        if not potential_players:
            # No potential matches - create new player
            player = Player(
                tracker_id=track_id,
                initial_embedding=track_embedding,
                initial_crops=track_data.crops,
                team=track_team
            )
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
                player.frame_first_seen = track_first_frame
                player.frame_last_seen = track_last_frame
                players.add(player)
                track_to_player[track_id] = player
                logger.debug(f"Created new player for track {track_id} "
                           f"(best similarity: {best_similarity:.3f} < {similarity_threshold})")
    
    # Log player association summary
    _log_player_association_summary(players)
    
    return players, track_to_player


def associate_tracks_to_players_globally(
    tracks_data: Dict[int, TrackData], 
    similarity_threshold: float = 0.96
) -> Tuple[set[Player], Dict[int, Player]]:
    """
    Associate tracks to players using global optimization.
    
    Args:
        tracks_data: Dictionary mapping track IDs to track data
        similarity_threshold: Minimum cosine similarity for association
        
    Returns:
        Tuple of (set of unique players, mapping from track ID to Player)
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        logger.warning("scipy not available, falling back to greedy approach")
        return associate_tracks_to_players_greedy(tracks_data, similarity_threshold)
    
    players: set[Player] = set()
    track_to_player: Dict[int, Player] = {}
    
    # Sort tracks by first frame to process them chronologically
    sorted_tracks = sorted(tracks_data.items(), key=lambda x: x[1].frame_first_seen)
    
    # Process tracks in batches for global optimization
    BATCH_SIZE = 10
    
    for batch_start in range(0, len(sorted_tracks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(sorted_tracks))
        batch_tracks = sorted_tracks[batch_start:batch_end]
        
        # Find potential players for all tracks in this batch
        all_potential_players = set()
        track_player_candidates = {}
        
        for track_id, track_data in batch_tracks:
            track_first_frame = track_data.frame_first_seen
            track_team = track_data.team
            
            # Find potential player matches
            potential_players = [
                player for player in players
                if player.frame_last_seen < track_first_frame and player.team == track_team
            ]
            
            track_player_candidates[track_id] = potential_players
            all_potential_players.update(potential_players)
        
        if all_potential_players:
            # Use Hungarian algorithm for optimal assignment within this batch
            batch_assignments = find_optimal_batch_assignment(
                batch_tracks, list(all_potential_players), similarity_threshold
            )
            
            # Apply assignments in chronological order
            for track_id, track_data in batch_tracks:
                assigned_player = batch_assignments.get(track_id)
                if assigned_player is not None:
                    # Associate with existing player
                    assigned_player.associated_tracker_ids.append(track_id)
                    assigned_player.crops.extend(track_data.crops)
                    assigned_player.frame_last_seen = track_data.frame_last_seen
                    
                    # Update player's embedding (average with existing)
                    if assigned_player.embeddings is not None:
                        assigned_player.embeddings = (assigned_player.embeddings + track_data.embedding) / 2.0
                    else:
                        assigned_player.embeddings = track_data.embedding
                    
                    track_to_player[track_id] = assigned_player
                    logger.debug(f"Globally assigned track {track_id} to player {assigned_player.id}")
                else:
                    # No assignment - create new player
                    player = Player(
                        tracker_id=track_id,
                        initial_embedding=track_data.embedding,
                        initial_crops=track_data.crops,
                        team=track_data.team
                    )
                    player.frame_first_seen = track_data.frame_first_seen
                    player.frame_last_seen = track_data.frame_last_seen
                    players.add(player)
                    track_to_player[track_id] = player
                    logger.debug(f"Created new player for track {track_id} (no assignment)")
        else:
            # No potential players, create new players for all tracks in batch
            for track_id, track_data in batch_tracks:
                player = Player(
                    tracker_id=track_id,
                    initial_embedding=track_data.embedding,
                    initial_crops=track_data.crops,
                    team=track_data.team
                )
                player.frame_first_seen = track_data.frame_first_seen
                player.frame_last_seen = track_data.frame_last_seen
                players.add(player)
                track_to_player[track_id] = player
                logger.debug(f"Created new player for track {track_id} (no potential players)")
    
    # Log player association summary
    _log_player_association_summary(players)
    
    return players, track_to_player


def associate_tracks_to_players_with_stitching(
    tracks_data: Dict[int, TrackData],
    multi_frame_detections: List,
    similarity_threshold: float = player_config.reid_similarity_threshold,
    stitch_similarity_threshold: float = track_stitching_config.stitch_similarity_threshold,
    max_time_gap: int = track_stitching_config.max_time_gap,
    appearance_weight: float = track_stitching_config.appearance_weight,
    temporal_weight: float = track_stitching_config.temporal_weight,
    motion_weight: float = track_stitching_config.motion_weight
) -> Tuple[set[Player], Dict[int, Player], List]:
    """
    Associate tracks to players using track stitching followed by global optimization.
    
    Args:
        tracks_data: Dictionary mapping track IDs to track data
        multi_frame_detections: List of detections for each frame to be updated
        similarity_threshold: Minimum cosine similarity for player association
        stitch_similarity_threshold: Minimum similarity for track stitching
        max_time_gap: Maximum frame gap allowed between tracklets for stitching
        appearance_weight: Weight for appearance similarity in stitching cost
        temporal_weight: Weight for temporal gap in stitching cost
        motion_weight: Weight for motion prediction in stitching cost
        
    Returns:
        Tuple of (set of unique players, mapping from track ID to Player, updated detections)
    """
    logger.info("Starting track association with stitching")
    
    # Check if stitching is enabled
    if not track_stitching_config.enable_stitching:
        logger.info("Track stitching is disabled - using standard global association")
        players, track_to_player = associate_tracks_to_players_globally(tracks_data, similarity_threshold)
        return players, track_to_player, multi_frame_detections
    
    # Step 1: Stitch fragmented tracklets
    logger.info("Step 1: Stitching fragmented tracklets")
    stitched_tracks, updated_multi_frame_detections, track_id_mapping = stitch_tracks(
        tracks_data=tracks_data,
        multi_frame_detections=multi_frame_detections,
        similarity_threshold=stitch_similarity_threshold,
        max_time_gap=max_time_gap,
        appearance_weight=appearance_weight,
        temporal_weight=temporal_weight,
        motion_weight=motion_weight
    )
    
    # Step 2: Associate stitched tracks to players
    logger.info("Step 2: Associating stitched tracks to players")
    players, stitched_track_to_player = associate_tracks_to_players_globally(
        tracks_data=stitched_tracks,
        similarity_threshold=similarity_threshold
    )
    
    # Step 3: Create mapping from original track IDs to players
    logger.info("Step 3: Creating mapping from original tracks to players")
    track_to_player = {}
    
    # Find which original tracks correspond to each stitched track
    for original_track_id in tracks_data.keys():
        found_player = None
        for stitched_track_id, player in stitched_track_to_player.items():
            stitched_track = stitched_tracks[stitched_track_id]
            # Check if this original track's data is part of the stitched track
            if (original_track_id == stitched_track_id or 
                _track_is_part_of_stitched_track(tracks_data[original_track_id], stitched_track)):
                found_player = player
                break
        
        if found_player:
            track_to_player[original_track_id] = found_player
        else:
            logger.warning(f"Could not find player for original track {original_track_id}")
    
    logger.info(f"Track association with stitching complete: {len(tracks_data)} original tracks -> "
                f"{len(stitched_tracks)} stitched tracks -> {len(players)} players")
    
    return players, track_to_player, updated_multi_frame_detections


def find_optimal_batch_assignment(
    batch_tracks: List[Tuple[int, TrackData]],
    potential_players: List[Player],
    similarity_threshold: float
) -> Dict[int, Optional[Player]]:
    """
    Find optimal assignment for a batch of tracks using Hungarian algorithm.
    
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
            if player.embeddings is not None:
                similarity = cosine_similarity(track_embedding, player.embeddings)
                if similarity >= similarity_threshold:
                    cost_matrix[i, j] = 1.0 - similarity  # Convert to cost
    
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


def _track_is_part_of_stitched_track(original_track: TrackData, stitched_track: TrackData) -> bool:
    """
    Check if an original track is part of a stitched track.
    
    Args:
        original_track: Original track data
        stitched_track: Stitched track data
        
    Returns:
        True if the original track is likely part of the stitched track
    """
    # Check if frame ranges overlap and teams match
    return (original_track.team == stitched_track.team and
            original_track.frame_first_seen >= stitched_track.frame_first_seen and
            original_track.frame_last_seen <= stitched_track.frame_last_seen)


def _log_player_association_summary(players: set[Player]):
    """
    Log a summary of player association results.
    
    Args:
        players: Set of Player objects to analyze
    """
    # Count how many tracks each player has
    tracks_per_player = [len(player.associated_tracker_ids) for player in players]
    
    # Count how many players have each number of tracks
    track_count_distribution = Counter(tracks_per_player)
    
    logger.info("Player association summary:")
    
    # Sort by number of tracks for consistent output
    for num_tracks in sorted(track_count_distribution.keys()):
        num_players = track_count_distribution[num_tracks]
        
        if num_players == 1:
            player_word = "player was"
            tracks_word = "track" if num_tracks == 1 else "tracks"
        else:
            player_word = "players were"
            tracks_word = "track" if num_tracks == 1 else "tracks"
        
        logger.info(f"  {num_players} {player_word} associated with {num_tracks} {tracks_word}")
    
    logger.info(f"Total: {len(players)} unique players from {sum(tracks_per_player)} tracks")
