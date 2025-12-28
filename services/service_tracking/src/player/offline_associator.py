"""
Offline player association - batch processing of all tracks after tracking completes.

This module provides a sophisticated algorithm for associating ByteTrack track IDs
with persistent player identities using:
- Team clustering via K-Means
- Per-team player discovery
- Spatial/velocity constraints
- Embedding bank for pose-invariant matching
- Global consistency enforcement
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from collections import defaultdict
import json

from .offline_config import OfflinePlayerConfig

logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """Information about a single track."""
    track_id: int
    start_frame: int
    end_frame: int
    first_bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    last_bbox: Tuple[float, float, float, float]
    embedding_mean: Optional[np.ndarray] = None
    embedding_variance: Optional[np.ndarray] = None
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    birth_type: str = 'unknown'  # 'edge' or 'mid'
    
    @property
    def first_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.first_bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def last_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.last_bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame
    
    @classmethod
    def from_frames_data(
        cls,
        track_id: int,
        frames: List[Tuple[int, Tuple[float, float, float, float]]]
    ) -> 'TrackInfo':
        """
        Create TrackInfo from list of (frame_id, bbox) tuples.
        
        Args:
            track_id: Track ID
            frames: List of (frame_id, bbox) sorted by frame_id
            
        Returns:
            TrackInfo instance
        """
        if not frames:
            raise ValueError(f"No frames provided for track {track_id}")
        
        frames_sorted = sorted(frames, key=lambda x: x[0])
        start_frame, first_bbox = frames_sorted[0]
        end_frame, last_bbox = frames_sorted[-1]
        
        return cls(
            track_id=track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            first_bbox=first_bbox,
            last_bbox=last_bbox,
        )


@dataclass 
class PlayerInfo:
    """Information about a discovered player."""
    player_id: int
    team_id: int
    track_ids: List[int] = field(default_factory=list)
    track_segments: List[Tuple[int, int, int]] = field(default_factory=list)  # (track_id, start, end)
    embedding_bank: List[np.ndarray] = field(default_factory=list)
    
    def add_track(self, track: TrackInfo):
        """Add a track to this player."""
        self.track_ids.append(track.track_id)
        self.track_segments.append((track.track_id, track.start_frame, track.end_frame))
        if track.embedding_mean is not None:
            self.embedding_bank.append(track.embedding_mean)
    
    def similarity_to(self, embedding: np.ndarray) -> float:
        """Compute max similarity to any embedding in the bank."""
        if not self.embedding_bank or embedding is None:
            return 0.0
        
        sims = [_cosine_similarity(embedding, e) for e in self.embedding_bank]
        return max(sims) if sims else 0.0
    
    def to_dict(self) -> dict:
        return {
            'player_id': self.player_id,
            'team_id': self.team_id,
            'track_ids': self.track_ids,
            'track_segments': [[int(t), int(s), int(e)] for t, s, e in self.track_segments],
        }


class OfflinePlayerAssociator:
    """
    Associates tracks with players in an offline (batch) manner.
    """
    
    def __init__(self, config: Optional[OfflinePlayerConfig] = None):
        self.config = config or OfflinePlayerConfig()
        self.tracks: Dict[int, TrackInfo] = {}
        self.players: Dict[int, PlayerInfo] = {}
        self.track_to_player: Dict[int, int] = {}
        self.next_player_id = 1
        self.frame_size: Tuple[int, int] = (1920, 1080)  # Default, will be set
    
    def load_data(
        self,
        tracks_path: str,
        embeddings_path: str,
        frame_size: Tuple[int, int] = (1920, 1080)
    ):
        """
        Load tracks and embeddings from files.
        
        Args:
            tracks_path: Path to tracks.json
            embeddings_path: Path to embeddings.pt
            frame_size: (width, height) of video frames
        """
        self.frame_size = frame_size
        
        # Load tracks
        with open(tracks_path, 'r') as f:
            tracks_data = json.load(f)
        
        # Build track info from frames
        track_frames: Dict[int, List[Tuple[int, Tuple]]] = defaultdict(list)
        
        for frame_data in tracks_data.get('frames', []):
            frame_id = frame_data['frame_id']
            for obj in frame_data.get('track_objects', []):
                track_id = obj['track_id']
                if track_id < 0:
                    continue  # Skip untracked detections
                
                bbox = tuple(obj.get('bbox', obj.get('tlbr', [0, 0, 0, 0])))
                track_frames[track_id].append((frame_id, bbox))
        
        # Create TrackInfo objects using classmethod
        for track_id, frames in track_frames.items():
            track = TrackInfo.from_frames_data(track_id, frames)
            track.birth_type = self._classify_birth(track)
            self.tracks[track_id] = track
        
        # Load embeddings
        import torch
        embeddings_data = torch.load(embeddings_path)
        
        for track_id, emb_data in embeddings_data.items():
            if track_id in self.tracks:
                if isinstance(emb_data, dict):
                    if 'mean' in emb_data:
                        mean = emb_data['mean']
                        if isinstance(mean, torch.Tensor):
                            mean = mean.cpu().numpy()
                        self.tracks[track_id].embedding_mean = mean.flatten()
                    if 'variance' in emb_data:
                        var = emb_data['variance']
                        if isinstance(var, torch.Tensor):
                            var = var.cpu().numpy()
                        self.tracks[track_id].embedding_variance = var.flatten()
                elif isinstance(emb_data, torch.Tensor):
                    self.tracks[track_id].embedding_mean = emb_data.cpu().numpy().flatten()
        
        logger.info(f"Loaded {len(self.tracks)} tracks with embeddings")
    
    def run(self) -> Dict:
        """
        Run the full offline association pipeline.
        
        Returns:
            Dictionary with player assignments
        """
        logger.info("Starting offline player association...")
        
        # Phase 1: Team Clustering
        self._cluster_teams()
        
        # Phase 1b: Infer missing teams (spatial propagation)
        self._infer_missing_teams()
        
        # Phase 2: Build constraint graph
        constraints = self._build_constraints()
        
        # Phase 3: Assign tracks to players per team
        for team_id in range(self.config.n_teams):
            team_tracks = [t for t in self.tracks.values() if t.team_id == team_id]
            self._assign_players_for_team(team_id, team_tracks, constraints)
        
        # Phase 4: Enforce global consistency
        self._enforce_global_consistency()
        
        # Phase 5: Merge fragmented players (spatial/temporal only)
        self._merge_fragmented_players()
        
        logger.info(f"Final Count: {len(self.players)} players from {len(self.tracks)} tracks")
        
        return self.export()
    
    def _cluster_teams(self):
        """Cluster tracks into teams using K-Means on embeddings."""
        logger.info("Phase 1: Clustering tracks into teams...")
        
        # Collect tracks with embeddings
        valid_tracks = [(t.track_id, t.embedding_mean) 
                       for t in self.tracks.values() 
                       if t.embedding_mean is not None]
        
        if len(valid_tracks) < self.config.n_teams:
            logger.warning("Not enough tracks with embeddings for team clustering")
            for track in self.tracks.values():
                track.team_id = 0
            return
        
        track_ids = [t[0] for t in valid_tracks]
        embeddings = np.array([t[1] for t in valid_tracks])
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=self.config.n_teams, n_init=10, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Compute confidence via silhouette score
        if len(embeddings) > self.config.n_teams:
            silhouette = silhouette_samples(embeddings, labels)
        else:
            silhouette = np.ones(len(embeddings))
        
        # Assign teams
        for i, track_id in enumerate(track_ids):
            self.tracks[track_id].team_id = int(labels[i])
        
        # Assign team -1 to tracks without embeddings (will be inferred later)
        for track in self.tracks.values():
            if track.team_id is None:
                track.team_id = -1
                
    def _infer_missing_teams(self):
        """Infer team IDs for tracks without embeddings based on spatial proximity."""
        logger.info("Phase 1b: Inferring missing teams...")
        
        sorted_tracks = sorted(self.tracks.values(), key=lambda t: t.start_frame)
        inferred_count = 0
        
        # Forward pass
        active_tracks = [] # List of (track, team_id)
        for track in sorted_tracks:
            # Update active tracks (remove old ones)
            active_tracks = [t for t in active_tracks if t[0].end_frame >= track.start_frame - 30]
            
            if track.team_id == -1:
                # Find nearest active track with a team
                best_team = -1
                min_dist = float('inf')
                
                for other, team in active_tracks:
                    if team == -1: continue
                    
                    dist = np.linalg.norm(np.array(track.first_center) - np.array(other.last_center))
                    if dist < min_dist:
                        min_dist = dist
                        best_team = team
                
                if best_team != -1 and min_dist < 200: # Reasonable jump distance
                    track.team_id = best_team
                    inferred_count += 1
            
            if track.team_id != -1:
                active_tracks.append((track, track.team_id))
        
        # Assign remaining to nearest centroid (fallback) or default to 0 if totally lost
        # Actually, let's just default remaining to 0 if inference fails, or distribute evenly?
        # For now, default remaining to 0 to ensure process continues
        remaining = 0
        for track in self.tracks.values():
            if track.team_id == -1:
                track.team_id = 0
                remaining += 1
                
        logger.info(f"Inferred teams for {inferred_count} tracks. Defaulted {remaining} to team 0.")
        
        # Log team distribution
        team_counts = defaultdict(int)
        for track in self.tracks.values():
            team_counts[track.team_id] += 1
        logger.info(f"Team distribution: {dict(team_counts)}")
    
    def _build_constraints(self) -> Dict[Tuple[int, int], bool]:
        """
        Build constraint matrix: can track i and j belong to same player?
        
        Returns:
            Dict[(track_i, track_j)] -> bool
        """
        logger.info("Building constraint graph...")
        constraints = {}
        
        track_list = list(self.tracks.values())
        for i, track_i in enumerate(track_list):
            for j, track_j in enumerate(track_list):
                if i >= j:
                    continue
                
                can_match = self._can_be_same_player(track_i, track_j)
                constraints[(track_i.track_id, track_j.track_id)] = can_match
                constraints[(track_j.track_id, track_i.track_id)] = can_match
        
        return constraints
    
    def _can_be_same_player(self, track_i: TrackInfo, track_j: TrackInfo) -> bool:
        """Check if two tracks can belong to the same player."""
        # Must be same team
        if track_i.team_id != track_j.team_id:
            return False
        
        # No temporal overlap
        if not (track_i.end_frame < track_j.start_frame or 
                track_j.end_frame < track_i.start_frame):
            return False
        
        # Spatial feasibility
        if not self._spatially_feasible(track_i, track_j):
            return False
        
        return True
    
    def _spatially_feasible(self, track_i: TrackInfo, track_j: TrackInfo) -> bool:
        """Check if a player could physically travel between track end and start."""
        # Determine which track ends first
        if track_i.end_frame < track_j.start_frame:
            end_pos = track_i.last_center
            start_pos = track_j.first_center
            gap_frames = track_j.start_frame - track_i.end_frame
        else:
            end_pos = track_j.last_center
            start_pos = track_i.first_center  
            gap_frames = track_i.start_frame - track_j.end_frame
        
        if gap_frames <= 0:
            return True  # Overlapping or adjacent
        
        # Calculate distance
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + 
                          (end_pos[1] - start_pos[1])**2)
        
        # Max possible travel
        max_speed_px = (self.config.max_speed_meters_per_second * 
                       self.config.pixels_per_meter / 
                       self.config.fps)
        max_travel = gap_frames * max_speed_px * self.config.velocity_margin
        
        return distance <= max_travel
    
    def _classify_birth(self, track: TrackInfo) -> str:
        """Classify track birth as 'edge' or 'mid'."""
        w, h = self.frame_size
        x, y = track.first_center
        
        edge_margin_x = w * self.config.edge_margin_ratio
        edge_margin_y = h * self.config.edge_margin_ratio
        field_top = h * self.config.field_top_ratio
        
        is_left_edge = x < edge_margin_x
        is_right_edge = x > w - edge_margin_x
        is_bottom_edge = y > h - edge_margin_y
        is_above_field = y < field_top  # Not really the field
        
        if is_left_edge or is_right_edge or is_bottom_edge:
            return 'edge'
        return 'mid'
    
    def _build_tracks_by_frame(self) -> Dict[int, List[TrackInfo]]:
        """Build mapping of frame_id -> List[TrackInfo] for tracks active in that frame."""
        tracks_by_frame = defaultdict(list)
        for track in self.tracks.values():
            for frame_id in range(track.start_frame, track.end_frame + 1):
                tracks_by_frame[frame_id].append(track)
        return tracks_by_frame
    
    def _find_anchor_frame(self, tracks_by_frame: Dict[int, List[TrackInfo]]) -> int:
        """
        Find frame with most detections. If tie, use best quality.
        Quality = avg embedding strength + avg duration of tracks in frame.
        """
        best_frame = None
        best_count = 0
        best_quality = 0.0
        
        for frame_id, tracks in tracks_by_frame.items():
            count = len(tracks)
            quality = sum(
                (1.0 if t.embedding_mean is not None else 0.0) + (t.duration / 100.0)
                for t in tracks
            ) / max(count, 1)
            
            if count > best_count or (count == best_count and quality > best_quality):
                best_frame = frame_id
                best_count = count
                best_quality = quality
        
        logger.info(f"Anchor frame: {best_frame} with {best_count} tracks (quality={best_quality:.2f})")
        return best_frame
    
    def _get_anchor_tracks(self, tracks: List[TrackInfo], frame_id: int) -> List[TrackInfo]:
        """Get all tracks active in the anchor frame."""
        return [t for t in tracks if t.start_frame <= frame_id <= t.end_frame]
    
    def _estimate_player_count(self, anchor_tracks: List[TrackInfo]) -> int:
        """
        Estimate optimal player count using silhouette analysis.
        """
        from sklearn.metrics import silhouette_score
        
        embeddings = np.array([t.embedding_mean for t in anchor_tracks 
                              if t.embedding_mean is not None])
        
        min_players = self.config.min_players_per_team
        max_players = self.config.max_players_per_team
        
        if len(embeddings) < min_players:
            logger.warning(f"Not enough embeddings ({len(embeddings)}) for player estimation")
            return self.config.default_players_per_team
        
        best_k = min_players
        best_score = -1
        
        for k in range(min_players, min(max_players + 1, len(embeddings))):
            clustering = AgglomerativeClustering(n_clusters=k)
            labels = clustering.fit_predict(embeddings)
            try:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        logger.info(f"Estimated {best_k} players (silhouette score={best_score:.3f})")
        return best_k
    
    def _assign_players_for_team(
        self, 
        team_id: int, 
        team_tracks: List[TrackInfo],
        constraints: Dict
    ):
        """Assign tracks to players within a team using similarity-first approach."""
        logger.info(f"Phase 3: Assigning players for team {team_id} "
                   f"({len(team_tracks)} tracks)")
        
        if not team_tracks:
            return
        
        # Step 1: Find anchor frame and tracks
        team_tracks_by_frame = {}
        for track in team_tracks:
            for frame_id in range(track.start_frame, track.end_frame + 1):
                if frame_id not in team_tracks_by_frame:
                    team_tracks_by_frame[frame_id] = []
                team_tracks_by_frame[frame_id].append(track)
        
        anchor_frame = self._find_anchor_frame(team_tracks_by_frame)
        anchor_tracks = self._get_anchor_tracks(team_tracks, anchor_frame)
        
        # Step 2: Estimate or use default player count
        if self.config.auto_estimate_player_count and len(anchor_tracks) >= self.config.min_players_per_team:
            n_players = self._estimate_player_count(anchor_tracks)
        else:
            n_players = self.config.default_players_per_team
        
        logger.info(f"Creating {n_players} initial players from {len(anchor_tracks)} anchor tracks")
        
        # Step 3: Seed initial players
        # Every track in the anchor frame MUST be a different player
        logger.info(f"Seeding {len(anchor_tracks)} players from anchor frame {anchor_frame}")
        for track in anchor_tracks:
            self._create_player_from_track(track, team_id)
        
        # Step 4: Build all valid (player, track) pairs with similarities
        candidates = self._build_assignment_candidates(team_tracks, constraints)
        
        # Step 5: Assign tracks by similarity (greedy from highest)
        self._assign_by_similarity(candidates, constraints)
        
        # Step 6: Handle orphan tracks
        # Any track not yet assigned that is long enough gets its own player
        assigned_tracks = set(self.track_to_player.keys())
        orphans_promoted = 0
        for track in team_tracks:
            if track.track_id not in assigned_tracks and track.duration > 15:
                self._create_player_from_track(track, team_id)
                orphans_promoted += 1
        
        if orphans_promoted > 0:
            logger.info(f"Promoted {orphans_promoted} significant orphan tracks to new players.")
    
    def _create_players_from_anchors(
        self,
        anchor_tracks: List[TrackInfo],
        n_players: int,
        team_id: int
    ):
        """Cluster anchor tracks into initial players."""
        # Get embeddings
        embeddings = np.array([t.embedding_mean for t in anchor_tracks 
                              if t.embedding_mean is not None])
        valid_anchors = [t for t in anchor_tracks if t.embedding_mean is not None]
        
        if len(embeddings) < n_players:
            # Fallback: create one player per anchor
            for track in anchor_tracks:
                self._create_player_from_track(track, team_id)
            return
        
        # Cluster with AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=n_players)
        labels = clustering.fit_predict(embeddings)
        
        # Create players and assign anchor tracks
        player_clusters = defaultdict(list)
        for track, label in zip(valid_anchors, labels):
            player_clusters[label].append(track)
        
        for cluster_tracks in player_clusters.values():
            player = PlayerInfo(
                player_id=self.next_player_id,
                team_id=team_id,
            )
            for track in cluster_tracks:
                player.add_track(track)
                track.player_id = player.player_id
                self.track_to_player[track.track_id] = player.player_id
            
            self.players[player.player_id] = player
            self.next_player_id += 1
    
    def _create_player_from_track(self, track: TrackInfo, team_id: int):
        """Create a new player with single track."""
        player = PlayerInfo(
            player_id=self.next_player_id,
            team_id=team_id,
        )
        player.add_track(track)
        track.player_id = player.player_id
        self.track_to_player[track.track_id] = player.player_id
        
        self.players[player.player_id] = player
        self.next_player_id += 1
    
    def _build_assignment_candidates(
        self,
        team_tracks: List[TrackInfo],
        constraints: Dict
    ) -> List[Tuple[PlayerInfo, TrackInfo, float]]:
        """Build all valid (player, track, similarity) tuples."""
        candidates = []
        assigned_tracks = set(self.track_to_player.keys())
        
        for track in team_tracks:
            if track.track_id in assigned_tracks:
                continue  # Already assigned to a player
            
            if track.embedding_mean is None:
                continue  # No embedding to match
            
            for player in self.players.values():
                if player.team_id != track.team_id:
                    continue  # Different team
                
                # Check constraints with all player's existing tracks
                valid = all(
                    constraints.get((track.track_id, existing_tid), True)
                    for existing_tid in player.track_ids
                )
                
                if not valid:
                    continue
                
                # Compute similarity
                sim = player.similarity_to(track.embedding_mean)
                
                # Apply birth-type boost
                if track.birth_type == 'mid':
                    sim *= self.config.mid_birth_priority_boost
                
                if sim >= self.config.similarity_threshold:
                    candidates.append((player, track, sim))
        
        return candidates
    
    def _assign_by_similarity(self, candidates: List[Tuple[PlayerInfo, TrackInfo, float]], constraints: Dict):
        """Assign tracks to players greedily from highest similarity, checking constraints dynamically."""
        # Sort by similarity descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        assigned_tracks = set(self.track_to_player.keys())
        
        for player, track, sim in candidates:
            if track.track_id in assigned_tracks:
                continue
            
            # CRITICAL: Dynamic conflict check against the player's CURRENT tracks
            conflict = False
            for existing_tid in player.track_ids:
                if not constraints.get((track.track_id, existing_tid), True):
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Assign!
            player.add_track(track)
            track.player_id = player.player_id
            self.track_to_player[track.track_id] = player.player_id
            assigned_tracks.add(track.track_id)
    
    def _enforce_global_consistency(self):
        """Ensure no player appears multiple times in same frame."""
        logger.info("Phase 4: Enforcing global consistency...")
        
        # 1. Identify conflicts
        player_frame_map = defaultdict(list) # player_id -> [frame_id]
        player_to_tracks = defaultdict(list) # player_id -> [track_info]
        
        for track in self.tracks.values():
            if track.player_id is not None:
                player_to_tracks[track.player_id].append(track)
        
        conflicts_resolved = 0
        
        for player_id, tracks in player_to_tracks.items():
            # Sort tracks by duration (keep the best ones)
            sorted_tracks = sorted(tracks, key=lambda t: t.duration, reverse=True)
            
            occupied_frames = set()
            for track in sorted_tracks:
                track_frames = set(range(track.start_frame, track.end_frame + 1))
                
                if track_frames & occupied_frames:
                    # Conflict! This track must be moved to a new player
                    conflicts_resolved += 1
                    
                    new_player = PlayerInfo(
                        player_id=self.next_player_id,
                        team_id=track.team_id,
                    )
                    # Deep update: remove from old player, add to new
                    old_player = self.players[player_id]
                    old_player.track_ids.remove(track.track_id)
                    # Filter track_segments
                    old_player.track_segments = [s for s in old_player.track_segments if s[0] != track.track_id]
                    
                    # Setup new player
                    new_player.add_track(track)
                    track.player_id = new_player.player_id
                    self.track_to_player[track.track_id] = new_player.player_id
                    
                    self.players[new_player.player_id] = new_player
                    self.next_player_id += 1
                else:
                    occupied_frames.update(track_frames)
        
        if conflicts_resolved > 0:
            logger.info(f"Resolved {conflicts_resolved} track conflicts by splitting players.")
            
    def _merge_fragmented_players(self):
        """
        Merge players that are likely the same person based on spatial/temporal feasibility.
        This handles cases where embeddings were missing but the physical path makes sense.
        """
        logger.info("Phase 5: Merging fragmented players...")
        merged_count = 0
        
        # Sort players by start frame
        sorted_players = sorted(
            self.players.values(), 
            key=lambda p: min(t[1] for t in p.track_segments) if p.track_segments else 0
        )
        
        # Simple greedy merge
        active_players = list(sorted_players)
        
        # We need a new validated list because we'll be removing items
        final_players = []
        
        while active_players:
            current = active_players.pop(0)
            
            # Try to merge with best subsequent player
            best_match = None
            best_gap = float('inf')
            
            # Look at potential matches (naive O(N^2) but N is small now)
            # Find the closest future player that fits constraints
            for candidate in active_players:
                if candidate.team_id != current.team_id:
                    continue
                
                # Check feasibility between LAST track of current and FIRST track of candidate
                last_track_id = current.track_ids[-1]
                first_track_id = candidate.track_ids[0]
                
                track_i = self.tracks[last_track_id]
                track_j = self.tracks[first_track_id]
                
                # Must be strictly after
                if track_j.start_frame <= track_i.end_frame:
                    continue
                
                if self._spatially_feasible(track_i, track_j):
                    gap = track_j.start_frame - track_i.end_frame
                    if gap < best_gap:
                        best_gap = gap
                        best_match = candidate
            
            if best_match and best_gap < 900: # 30 seconds max gap for blind merge
                # Merge candidate INTO current
                for tid in best_match.track_ids:
                    # Update mappings
                    self.track_to_player[tid] = current.player_id
                    current.track_ids.append(tid)
                    
                    # Update segments
                    for seg in best_match.track_segments:
                        if seg[0] == tid:
                            current.track_segments.append(seg)
                    
                    # Add embeddings
                    track = self.tracks[tid]
                    if track.embedding_mean is not None:
                        current.embedding_bank.append(track.embedding_mean)
                
                # Sort tracks in current player
                current.track_segments.sort(key=lambda x: x[1])
                
                # Remove best_match from processing list
                active_players.remove(best_match)
                # Remove from main dict
                del self.players[best_match.player_id]
                
                # Re-add current to list to potentialy merge again!
                active_players.insert(0, current)
                merged_count += 1
            else:
                final_players.append(current)
        
        if merged_count > 0:
            logger.info(f"Merged {merged_count} fragmented player identities.")
    
    def export(self) -> Dict:
        """Export results to dictionary."""
        teams = defaultdict(list)
        for player in self.players.values():
            teams[player.team_id].append(player.player_id)
        
        return {
            'teams': {
                str(team_id): {'player_ids': pids}
                for team_id, pids in teams.items()
            },
            'players': {
                str(pid): player.to_dict()
                for pid, player in self.players.items()
            },
            'track_to_player': {
                str(tid): {'player_id': pid, 'team_id': self.players[pid].team_id}
                for tid, pid in self.track_to_player.items()
                if pid in self.players
            },
            'statistics': {
                'total_players': len(self.players),
                'total_tracks': len(self.tracks),
                'teams': len(teams),
            }
        }
    
    def save(self, output_path: str):
        """Save results to JSON file."""
        data = self.export()
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved player associations to {output_path}")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))
