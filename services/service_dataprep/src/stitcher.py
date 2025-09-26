from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
import logging
import numpy as np
import networkx as nx

from shared_libs.common.detection_format import Detections
from shared_libs.config import logging_config  # Import logging configuration

# Set up logger
logger = logging.getLogger(__name__)

class EdgeType(Enum):
    """Types of relationships between tracks."""
    SAME_PLAYER = "same_player"
    DIFFERENT_PLAYER = "different_player"  
    TEMPORAL_CONFLICT = "temporal_conflict"  # Auto-detected overlaps
    SKIPPED = "skipped"  # User couldn't decide, deferred for later
    UNKNOWN = "unknown"

class TrackStitcher:
    """
    Manages the semi-automated process of stitching player tracks together.

    This class provides an API to get pairs of unverified track groups for manual
    verification and handles the merging based on user input. It uses a graph-based
    approach to track relationships between track groups.

    Supports a progressive refinement workflow where unclear pairs can be skipped
    during the first pass and revisited with enhanced context in a second pass.

    Example Usage:
        ```python
        # Initialize stitcher
        stitcher = TrackStitcher(detections=detections_obj)

        # First pass - skip unclear pairs
        while True:
            result = stitcher.get_pair_for_verification()

            if result["status"] == "pending_verification":
                # Display images for result["group1_id"] and result["group2_id"]
                # (retrieve images externally based on group IDs)

                # Get user decision
                decision = input("Are these the same player? (same/different/skip): ")
                stitcher.respond(decision)

            elif result["status"] == "second_pass_ready":
                print(f"First pass complete! {result['skipped_count']} pairs skipped")
                break
            elif result["status"] == "complete":
                print("All verification complete!")
                break

        # Save progress for potential resume later
        stitcher.save_graph("checkpoint.graphml")

        # Second pass - revisit skipped pairs with enhanced context
        if stitcher.start_second_pass():
            while True:
                result = stitcher.get_pair_for_verification()
                
                if result["status"] == "pending_verification":
                    print(f"Context for group1: {result['group1'].get('context', 'N/A')}")
                    print(f"Context for group2: {result['group2'].get('context', 'N/A')}")
                    
                    decision = input("Same player? (same/different/skip): ")
                    stitcher.respond(decision)
                elif result["status"] == "complete":
                    break

        # To resume from a saved checkpoint:
        import networkx as nx
        saved_graph = nx.read_graphml("checkpoint.graphml")
        stitcher = TrackStitcher(detections=detections_obj, existing_graph=saved_graph)
        # Continue verification from where you left off
        ```

    Key Features:
        - Skip unclear pairs without blocking progress: respond("skip")
        - Progressive refinement with enhanced context in second pass
        - Graph-based relationship tracking and visualization
        - Automatic temporal conflict detection
        - Rich progress analytics and inconsistency detection
        - Returns group IDs only (image handling is external)
        - Visualization returns PIL.Image objects for flexible display/saving
        - Resume interrupted verification from saved graph checkpoints
    """

    def __init__(self, detections: Detections, existing_graph: Optional[nx.Graph] = None):
        """
        Initializes the TrackStitcher.

        Args:
            detections (Detections): Detections object containing track information.
            existing_graph (Optional[nx.Graph]): Existing graph to resume from. If provided,
                                               the stitcher will load the previous state and
                                               allow continuing verification.
        """
        logger.info("Initializing TrackStitcher...")
        self.detections = detections

        # 1. Pre-process supervision data for efficient lookups
        self._track_to_frames: Dict[int, np.ndarray] = self._invert_supervision_data(self.detections)
        
        # 2. Discover all track IDs using detections-derived track IDs
        all_track_ids = sorted(self._track_to_frames.keys())
        logger.info(f"Found {len(all_track_ids)} tracks (from detections).")

        if existing_graph is not None:
            logger.info("Loading existing graph and reconstructing state...")
            self._load_existing_graph(existing_graph, all_track_ids)
        else:
            # 3. Initialize player groups. Initially, each track is its own group.
            # The key is the representative ID for the group (we use the smallest track_id).
            self.player_groups: Dict[int, Set[int]] = {tid: {tid} for tid in all_track_ids}

            # 4. Initialize graph representation for advanced relationship tracking
            self.track_graph = nx.Graph()
            self.track_graph.add_nodes_from(all_track_ids)
            self._populate_temporal_conflicts()
        
        # 5. State for the comparison iterator
        self._comparison_keys: List[int] = []
        self._i = 0
        self._j = 1
        self._last_proposed_pair: Optional[Tuple[int, int]] = None
        
        # 6. Verification mode tracking
        self._verification_mode = "normal"  # "normal", "second_pass", or "skipped_only"
        
        logger.info("Initialization complete. Ready for verification.")

    def _load_existing_graph(self, existing_graph: nx.Graph, expected_track_ids: List[int]):
        """
        Load an existing graph and reconstruct the stitcher state.
        
        Args:
            existing_graph: The saved graph to load
            expected_track_ids: Track IDs that should exist based on detections
        """
        # Handle potential string node IDs from GraphML format
        graph_nodes = list(existing_graph.nodes())
        if graph_nodes and isinstance(graph_nodes[0], str):
            # Convert string node IDs back to integers
            mapping = {str(node_id): node_id for node_id in expected_track_ids}
            existing_graph = nx.relabel_nodes(existing_graph, mapping)
        
        # Convert string relationship values back to EdgeType enums
        for u, v, data in existing_graph.edges(data=True):
            if 'relationship' in data and isinstance(data['relationship'], str):
                try:
                    data['relationship'] = EdgeType(data['relationship'])
                except ValueError:
                    # If it's not a valid EdgeType, leave as string or set to UNKNOWN
                    data['relationship'] = EdgeType.UNKNOWN
        
        # Validate that the graph contains the expected tracks
        graph_track_ids = set(existing_graph.nodes())
        expected_set = set(expected_track_ids)
        
        if not expected_set.issubset(graph_track_ids):
            missing = expected_set - graph_track_ids
            raise ValueError(f"Existing graph is missing tracks: {sorted(missing)}")
        
        if graph_track_ids - expected_set:
            extra = graph_track_ids - expected_set
            logger.warning(f"Graph contains extra tracks not in detections: {sorted(extra)}")
        
        # Set the graph
        self.track_graph = existing_graph.copy()
        
        # Ensure all expected tracks are in the graph
        for track_id in expected_track_ids:
            if track_id not in self.track_graph:
                self.track_graph.add_node(track_id)
        
        # Reconstruct player groups from connected components (SAME_PLAYER relationships)
        components = self._get_connected_components()
        self.player_groups = {}
        
        for component in components:
            # Use the smallest track ID as the group representative
            group_id = min(component)
            self.player_groups[group_id] = component
        
        # Ensure temporal conflicts are present (re-run detection if needed)
        existing_temporal_edges = 0
        for _, _, data in self.track_graph.edges(data=True):
            if data.get('relationship') == EdgeType.TEMPORAL_CONFLICT:
                existing_temporal_edges += 1
        
        if existing_temporal_edges == 0:
            logger.info("Re-detecting temporal conflicts...")
            self._populate_temporal_conflicts()
        else:
            logger.info(f"Found {existing_temporal_edges} existing temporal conflict relationships")
        
        verified_tracks = sum(len(group) for group in self.player_groups.values())
        logger.info(f"Reconstructed {len(self.player_groups)} player groups with {verified_tracks} tracks")


    def _invert_supervision_data(self, detections: Detections) -> Dict[int, np.ndarray]:
        """Converts frame->tracks mapping to track->frames for faster lookups.
        
        Returns track_id -> sorted numpy array of frame indices for efficient operations.
        """
        try:
            frame_indices = detections.data['frame_index']
            track_ids = detections.data['tracker_id']
        except KeyError as e:
            raise ValueError(f"Detections data must include '{e.args[0]}' for each detection.")

        # Convert to numpy arrays with appropriate dtypes for memory efficiency
        frame_indices = np.asarray(frame_indices, dtype=np.int32)
        track_ids = np.asarray(track_ids, dtype=np.int32)

        track_to_frames = {}
        
        # Get unique track IDs - much smaller than total detections (400 vs 2M)
        unique_tracks = np.unique(track_ids)

        # For each unique track, find all frames it appears in
        for track_id in unique_tracks:
            # Boolean mask for current track - vectorized operation
            mask = track_ids == track_id
            # Keep as sorted numpy array for efficient intersection operations
            track_frames = np.sort(frame_indices[mask])
            track_to_frames[int(track_id)] = track_frames

        return track_to_frames

    def _populate_temporal_conflicts(self):
        """Pre-populate graph with temporal conflict edges (tracks that overlap in time)."""
        logger.info("Detecting temporal conflicts...")
        
        all_track_ids = list(self.track_graph.nodes())
        conflicts_found = 0
        
        for i, track1 in enumerate(all_track_ids):
            for track2 in all_track_ids[i+1:]:
                if self._tracks_overlap_in_time(track1, track2):
                    self.track_graph.add_edge(track1, track2, relationship=EdgeType.TEMPORAL_CONFLICT)
                    conflicts_found += 1
        
        logger.info(f"Found {conflicts_found} temporal conflicts (auto-rejected pairs)")

    def _tracks_overlap_in_time(self, track1_id: int, track2_id: int) -> bool:
        """Check if two individual tracks overlap in time."""
        frames1 = self._track_to_frames.get(track1_id, np.array([], dtype=np.int32))
        frames2 = self._track_to_frames.get(track2_id, np.array([], dtype=np.int32))
        
        if len(frames1) == 0 or len(frames2) == 0:
            return False
            
        # Use numpy's efficient intersect1d for sorted arrays
        intersection = np.intersect1d(frames1, frames2, assume_unique=True)
        return len(intersection) > 0

    def _groups_overlap_in_time(self, group1_id: int, group2_id: int) -> bool:
        """Checks if two player groups ever appear in the same frame."""
        group1_tracks = self.player_groups[group1_id]
        group2_tracks = self.player_groups[group2_id]

        # Collect all frame arrays for each group
        group1_frame_arrays = [self._track_to_frames.get(tid, np.array([], dtype=np.int32)) 
                              for tid in group1_tracks]
        group2_frame_arrays = [self._track_to_frames.get(tid, np.array([], dtype=np.int32)) 
                              for tid in group2_tracks]
        
        # Concatenate frames for each group (sorted arrays stay sorted when concatenated)
        group1_frames = np.concatenate(group1_frame_arrays) if group1_frame_arrays else np.array([], dtype=np.int32)
        group2_frames = np.concatenate(group2_frame_arrays) if group2_frame_arrays else np.array([], dtype=np.int32)
        
        # Use numpy's efficient intersect1d - works well with sorted arrays
        intersection = np.intersect1d(group1_frames, group2_frames, assume_unique=False)
        return len(intersection) > 0

    def get_pair_for_verification(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Gets the next pair of track groups for the user to verify.

        Args:
            mode: Verification mode - None (use current), "normal", "second_pass", or "skipped_only"

        Returns:
            A dictionary with the status and data. Possible statuses:
            - 'pending_verification': User input is needed. 'group1_id' and 'group2_id' keys
                                      contain the group IDs for external image retrieval.
            - 'complete': All possible pairs have been verified.
            - 'second_pass_ready': Normal pass complete, ready for second pass.
        """
        if mode is not None:
            self._verification_mode = mode
            
        self._last_proposed_pair = None
        
        if self._verification_mode == "skipped_only":
            return self._get_skipped_pair_for_verification()
        elif self._verification_mode == "second_pass":
            return self._get_second_pass_pair()
        else:  # normal mode
            return self._get_normal_pair_for_verification()

    def _get_normal_pair_for_verification(self) -> Dict[str, Any]:
        """Get pairs for normal verification (skip already processed pairs)."""
        self._comparison_keys = sorted(self.player_groups.keys())

        while self._i < len(self._comparison_keys) - 1:
            while self._j < len(self._comparison_keys):
                group1_id = self._comparison_keys[self._i]
                group2_id = self._comparison_keys[self._j]

                # Skip if already processed (has any edge between groups)
                if self._groups_have_any_relationship(group1_id, group2_id):
                    self._j += 1
                    continue

                # Automatic rejection: if they appear in the same frame, skip.
                if self._groups_overlap_in_time(group1_id, group2_id):
                    # Record temporal conflict and skip
                    self._record_group_verification(group1_id, group2_id, EdgeType.TEMPORAL_CONFLICT)
                    self._j += 1
                    continue

                # We have a candidate pair for manual verification.
                self._last_proposed_pair = (group1_id, group2_id)

                return {
                    "status": "pending_verification",
                    "mode": "normal",
                    "group1_id": group1_id,
                    "group2_id": group2_id
                }

            # Move to the next primary group to compare
            self._i += 1
            self._j = self._i + 1 # Reset secondary group

        # Normal pass complete - check if there are skipped pairs
        skipped_count = len(self.get_skipped_pairs())
        if skipped_count > 0:
            return {
                "status": "second_pass_ready", 
                "message": f"✅ Normal verification complete! {skipped_count} skipped pairs available for second pass.",
                "skipped_count": skipped_count
            }
        else:
            return {"status": "complete", "message": "✅ All tracks have been assigned to players."}

    def _groups_have_any_relationship(self, group1_id: int, group2_id: int) -> bool:
        """Check if two groups have any established relationship (not unknown)."""
        group1_tracks = self.player_groups[group1_id]
        group2_tracks = self.player_groups[group2_id]
        
        # Check if any tracks between the groups have edges
        for track1 in group1_tracks:
            for track2 in group2_tracks:
                if self.track_graph.has_edge(track1, track2):
                    return True
        return False

    def respond(self, decision: str):
        """
        Provides the user's verification result for the last proposed pair.

        Args:
            decision (str): User's decision - "same", "different", or "skip"
        """
        if self._last_proposed_pair is None:
            raise RuntimeError("respond() called before a pair was proposed by get_pair_for_verification().")

        if decision not in ["same", "different", "skip"]:
            raise ValueError(f"Invalid decision '{decision}'. Must be 'same', 'different', or 'skip'.")

        group1_id, group2_id = self._last_proposed_pair
        
        if decision == "skip":
            # Record skip in graph and move to next pair
            self._record_group_verification(group1_id, group2_id, EdgeType.SKIPPED)
            logger.info(f"Pair skipped for later review (groups {group1_id} and {group2_id})")
            self._j += 1
            
        elif decision == "same":
            # Record verification in graph
            self._record_group_verification(group1_id, group2_id, EdgeType.SAME_PLAYER)
            
            logger.info(f"Merging group {group2_id} into group {group1_id}...")
            
            # Always merge into the group with the smaller ID to maintain consistency
            if group1_id > group2_id:
                group1_id, group2_id = group2_id, group1_id
            
            # Merge the track IDs from group2 into group1
            self.player_groups[group1_id].update(self.player_groups[group2_id])
            
            # Remove the now-merged group2
            del self.player_groups[group2_id]

            # After a merge, we must restart the comparison from the beginning
            # to compare the new, larger group against all others.
            self._i = 0
            self._j = 1
            logger.info("Merge complete. Restarting comparison scan.")
            
        elif decision == "different":
            # Record verification in graph
            self._record_group_verification(group1_id, group2_id, EdgeType.DIFFERENT_PLAYER)
            logger.info("Groups marked as different. Continuing...")
            self._j += 1
            
        self._last_proposed_pair = None

    def _get_skipped_pair_for_verification(self) -> Dict[str, Any]:
        """Get the next skipped pair for re-verification."""
        skipped_pairs = self.get_skipped_pairs()

        if not skipped_pairs:
            return {"status": "complete", "message": "✅ All skipped pairs have been processed."}

        # Get the first skipped pair
        group1_id, group2_id = skipped_pairs[0]
        self._last_proposed_pair = (group1_id, group2_id)

        return {
            "status": "pending_verification",
            "mode": "skipped_only",
            "group1_id": group1_id,
            "group2_id": group2_id,
            "context": f"Previously skipped pair ({len(skipped_pairs)} skipped pairs remaining)"
        }

    def _get_second_pass_pair(self) -> Dict[str, Any]:
        """Get pairs for second pass verification (enhanced context for skipped pairs)."""
        skipped_pairs = self.get_skipped_pairs()

        if not skipped_pairs:
            return {"status": "complete", "message": "✅ All skipped pairs have been resolved."}

        # For now, same as skipped_only but with enhanced context messaging
        group1_id, group2_id = skipped_pairs[0]
        self._last_proposed_pair = (group1_id, group2_id)

        # Get context about what groups these have been merged with
        group1_context = self._get_group_context(group1_id)
        group2_context = self._get_group_context(group2_id)

        return {
            "status": "pending_verification",
            "mode": "second_pass",
            "group1_id": group1_id,
            "group2_id": group2_id,
            "group1_context": group1_context,
            "group2_context": group2_context,
            "message": f"Second pass: Enhanced context available ({len(skipped_pairs)} skipped pairs remaining)"
        }

    def _get_group_context(self, group_id: int) -> str:
        """Get contextual information about a group's verification history."""
        tracks_in_group = self.player_groups[group_id]
        context_info = []
        
        if len(tracks_in_group) > 1:
            context_info.append(f"Merged from {len(tracks_in_group)} original tracks: {sorted(tracks_in_group)}")
        
        # Count relationships this group has
        same_count = 0
        diff_count = 0
        
        for other_group_id in self.player_groups.keys():
            if other_group_id == group_id:
                continue
            
            other_tracks = self.player_groups[other_group_id]
            for track1 in tracks_in_group:
                for track2 in other_tracks:
                    if self.track_graph.has_edge(track1, track2):
                        rel = self.track_graph[track1][track2]['relationship']
                        if rel == EdgeType.SAME_PLAYER:
                            same_count += 1
                        elif rel == EdgeType.DIFFERENT_PLAYER:
                            diff_count += 1
                        break  # Only count once per group pair
                else:
                    continue
                break
        
        if same_count > 0 or diff_count > 0:
            context_info.append(f"Verified relationships: {same_count} same, {diff_count} different")
        
        return "; ".join(context_info) if context_info else "No additional context"

    def get_skipped_pairs(self) -> List[Tuple[int, int]]:
        """Get all pairs that were skipped during verification."""
        skipped_pairs = []
        
        # Find all group pairs with skipped edges
        processed_pairs = set()
        
        for group1_id in self.player_groups.keys():
            for group2_id in self.player_groups.keys():
                if group1_id >= group2_id:
                    continue
                    
                pair = (group1_id, group2_id)
                if pair in processed_pairs:
                    continue
                    
                # Check if this group pair has any skipped edges
                group1_tracks = self.player_groups[group1_id]
                group2_tracks = self.player_groups[group2_id]
                
                has_skipped = False
                for track1 in group1_tracks:
                    for track2 in group2_tracks:
                        if (self.track_graph.has_edge(track1, track2) and 
                            self.track_graph[track1][track2]['relationship'] == EdgeType.SKIPPED):
                            has_skipped = True
                            break
                    if has_skipped:
                        break
                
                if has_skipped:
                    skipped_pairs.append(pair)
                    processed_pairs.add(pair)
        
        return skipped_pairs

    def start_second_pass(self) -> bool:
        """Start the second pass verification for skipped pairs."""
        skipped_count = len(self.get_skipped_pairs())
        if skipped_count == 0:
            logger.info("No skipped pairs found. Verification is complete!")
            return False
        
        logger.info(f"Starting second pass verification for {skipped_count} skipped pairs...")
        logger.info("Tip: You now have more context from merged groups to help with decisions.")
        self._verification_mode = "second_pass"
        return True

    def _record_group_verification(self, group1_id: int, group2_id: int, relationship: EdgeType):
        """Record verification result in the graph for all track pairs between groups."""
        group1_tracks = self.player_groups[group1_id]
        group2_tracks = self.player_groups[group2_id]
        
        # Add edges between all tracks in the two groups
        for track1 in group1_tracks:
            for track2 in group2_tracks:
                if self.track_graph.has_edge(track1, track2):
                    # Update existing edge
                    self.track_graph[track1][track2]['relationship'] = relationship
                else:
                    # Add new edge
                    self.track_graph.add_edge(track1, track2, relationship=relationship)


    def get_verification_progress(self) -> Dict[str, Any]:
        """Get detailed analytics about verification progress."""
        total_nodes = self.track_graph.number_of_nodes()
        total_possible_pairs = total_nodes * (total_nodes - 1) // 2
        
        # Count edges by type
        edge_counts = {edge_type: 0 for edge_type in EdgeType}
        for _, _, data in self.track_graph.edges(data=True):
            relationship = data.get('relationship', EdgeType.UNKNOWN)
            edge_counts[relationship] += 1
        
        # Calculate remaining pairs (those without edges)
        total_edges = self.track_graph.number_of_edges()
        remaining_pairs = total_possible_pairs - total_edges
        
        progress_percentage = (total_edges / total_possible_pairs * 100) if total_possible_pairs > 0 else 100
        
        return {
            'total_tracks': total_nodes,
            'total_possible_pairs': total_possible_pairs,
            'verified_pairs': edge_counts[EdgeType.SAME_PLAYER] + edge_counts[EdgeType.DIFFERENT_PLAYER],
            'same_player_pairs': edge_counts[EdgeType.SAME_PLAYER],
            'different_player_pairs': edge_counts[EdgeType.DIFFERENT_PLAYER],
            'temporal_conflicts': edge_counts[EdgeType.TEMPORAL_CONFLICT],
            'skipped_pairs': edge_counts[EdgeType.SKIPPED],
            'remaining_pairs': remaining_pairs,
            'progress_percentage': round(progress_percentage, 1),
            'current_player_groups': len(self.player_groups),
            'verification_mode': self._verification_mode
        }

    def detect_inconsistencies(self) -> List[str]:
        """Detect logical inconsistencies in verification results."""
        issues = []
        
        # Get connected components using only SAME_PLAYER edges
        same_player_graph = nx.Graph()
        same_player_graph.add_nodes_from(self.track_graph.nodes())
        
        for u, v, data in self.track_graph.edges(data=True):
            if data['relationship'] == EdgeType.SAME_PLAYER:
                same_player_graph.add_edge(u, v)
        
        # Check for inconsistencies within components
        for component in nx.connected_components(same_player_graph):
            component = list(component)
            for i, track1 in enumerate(component):
                for track2 in component[i+1:]:
                    if (self.track_graph.has_edge(track1, track2) and 
                        self.track_graph[track1][track2]['relationship'] == EdgeType.DIFFERENT_PLAYER):
                        issues.append(f"Inconsistency: tracks {track1} and {track2} are connected via SAME_PLAYER but directly marked DIFFERENT_PLAYER")
        
        return issues

    def get_verification_candidates(self, prioritize_by_size: bool = True) -> List[Tuple[int, int]]:
        """Get list of track pairs that still need verification, optionally prioritized."""
        candidates = []
        
        # Find all pairs without edges (unverified)
        all_tracks = list(self.track_graph.nodes())
        for i, track1 in enumerate(all_tracks):
            for track2 in all_tracks[i+1:]:
                if not self.track_graph.has_edge(track1, track2):
                    candidates.append((track1, track2))
        
        if prioritize_by_size and candidates:
            # Prioritize pairs that could merge larger components
            same_player_components = self._get_connected_components()
            
            # Score each candidate by the size of components it could merge
            def score_candidate(pair):
                track1, track2 = pair
                comp1_size = len(self._get_component_for_track(track1, same_player_components))
                comp2_size = len(self._get_component_for_track(track2, same_player_components))
                return comp1_size * comp2_size  # Larger product = higher impact merge
            
            candidates.sort(key=score_candidate, reverse=True)
        
        return candidates

    def _get_connected_components(self) -> List[Set[int]]:
        """Get connected components using only SAME_PLAYER edges."""
        same_player_graph = nx.Graph()
        same_player_graph.add_nodes_from(self.track_graph.nodes())
        
        for u, v, data in self.track_graph.edges(data=True):
            if data['relationship'] == EdgeType.SAME_PLAYER:
                same_player_graph.add_edge(u, v)
        
        return [set(component) for component in nx.connected_components(same_player_graph)]

    def _get_component_for_track(self, track_id: int, components: List[Set[int]]) -> Set[int]:
        """Find which component contains the given track."""
        for component in components:
            if track_id in component:
                return component
        return {track_id}  # Isolated track

    def visualize_graph(self, show_labels: bool = True):
        """
        Create visual representation of track relationships and return as a PIL.Image object.

        Args:
            show_labels: Whether to show track IDs as node labels in the graph.

        Returns:
            PIL.Image object containing the graph visualization, or None if visualization fails.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            import matplotlib.cm as cm
            from io import BytesIO
            from PIL import Image

            plt.figure(figsize=(15, 10))

            # Create layout
            pos = nx.spring_layout(self.track_graph, k=3, iterations=50)

            # Separate edges by type for different colors
            edge_colors = []
            for u, v, data in self.track_graph.edges(data=True):
                relationship = data.get('relationship', EdgeType.UNKNOWN)
                if relationship == EdgeType.SAME_PLAYER:
                    edge_colors.append('green')
                elif relationship == EdgeType.DIFFERENT_PLAYER:
                    edge_colors.append('red')
                elif relationship == EdgeType.TEMPORAL_CONFLICT:
                    edge_colors.append('orange')
                elif relationship == EdgeType.SKIPPED:
                    edge_colors.append('purple')
                else:
                    edge_colors.append('gray')

            # Color nodes by connected component (same player group)
            components = self._get_connected_components()
            all_nodes = list(self.track_graph.nodes())
            node_colors = ['lightblue'] * len(all_nodes)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            if len(components) > 1:
                for i, component in enumerate(components):
                    color = colors[i % len(colors)]
                    for track in component:
                        if track in all_nodes:
                            node_idx = all_nodes.index(track)
                            node_colors[node_idx] = color

            nx.draw(self.track_graph, pos,
                    node_color=node_colors,
                    edge_color=edge_colors,
                    with_labels=show_labels,
                    node_size=300,
                    font_size=8,
                    font_weight='bold')

            legend_elements = [
                Line2D([0], [0], color='green', lw=2, label='Same Player'),
                Line2D([0], [0], color='red', lw=2, label='Different Player'),
                Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Temporal Conflict'),
                Line2D([0], [0], color='purple', lw=2, linestyle='-.', label='Skipped'),
                Line2D([0], [0], color='gray', lw=2, linestyle=':', label='Unknown')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title("Track Relationship Graph\n(Node colors represent player groups)")
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            image = Image.open(buf)
            return image

        except ImportError:
            print("⚠️  Matplotlib or PIL not available. Install with: pip install matplotlib pillow")
            return None
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return None

    def get_graph(self) -> nx.Graph:
        """
        Get access to the underlying NetworkX graph for external analysis or saving.

        Returns:
            NetworkX Graph object containing track relationships.
        """
        return self.track_graph

    def save_graph(self, filepath: str, format: str = "graphml") -> bool:
        """
        Save the track relationship graph to a file.

        Args:
            filepath: Path where to save the graph file.
            format: File format - "graphml" (default), "gml", "pickle", "json", etc.

        Returns:
            True if save was successful, False otherwise.
        """
        try:
            if format.lower() == "graphml":
                # Create a copy with string relationship values for GraphML compatibility
                graph_copy = self.track_graph.copy()
                for u, v, data in graph_copy.edges(data=True):
                    if 'relationship' in data and isinstance(data['relationship'], EdgeType):
                        data['relationship'] = data['relationship'].value
                nx.write_graphml(graph_copy, filepath)
            elif format.lower() == "gml":
                # Create a copy with string relationship values for GML compatibility
                graph_copy = self.track_graph.copy()
                for u, v, data in graph_copy.edges(data=True):
                    if 'relationship' in data and isinstance(data['relationship'], EdgeType):
                        data['relationship'] = data['relationship'].value
                nx.write_gml(graph_copy, filepath)
            elif format.lower() == "pickle":
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(self.track_graph, f)
            elif format.lower() == "json":
                # Convert to node-link format for JSON, handling Enum serialization
                graph_data = nx.node_link_data(self.track_graph)
                # Convert Enum values to strings for JSON serialization
                for link in graph_data.get('links', []):
                    if 'relationship' in link and isinstance(link['relationship'], EdgeType):
                        link['relationship'] = link['relationship'].value
                import json
                with open(filepath, 'w') as f:
                    json.dump(graph_data, f, indent=2)
            else:
                print(f"❌ Unsupported format: {format}. Supported: graphml, gml, pickle, json")
                return False

            print(f"✅ Graph saved to {filepath} in {format} format")
            return True

        except Exception as e:
            print(f"❌ Error saving graph: {e}")
            return False

    def export_graph_data(self) -> Dict[str, Any]:
        """
        Export graph data as a dictionary for serialization or external processing.

        Returns:
            Dictionary containing nodes, edges, and metadata.
        """
        # Get connected components (player groups)
        components = self._get_connected_components()

        # Convert edges to serializable format
        edges_data = []
        for u, v, data in self.track_graph.edges(data=True):
            edges_data.append({
                'track1': u,
                'track2': v,
                'relationship': data.get('relationship', EdgeType.UNKNOWN).value
            })

        # Convert components to serializable format
        components_data = [sorted(list(comp)) for comp in components]

        return {
            'nodes': list(self.track_graph.nodes()),
            'edges': edges_data,
            'player_groups': components_data,
            'player_groups_dict': {min(group): sorted(list(group)) for group in components_data},
            'metadata': {
                'total_tracks': self.track_graph.number_of_nodes(),
                'total_relationships': len(edges_data),
                'player_count': len(components),
                'verification_mode': self._verification_mode
            }
        }