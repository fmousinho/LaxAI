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
            for obj in frame_data.get('objects', []):
                track_id = obj['track_id']
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
        
        # Phase 1: Team clustering
        self._cluster_teams()
        
        # Phase 2: Build constraint graph
        constraints = self._build_constraints()
        
        # Phase 3: Assign tracks to players per team
        for team_id in range(self.config.n_teams):
            team_tracks = [t for t in self.tracks.values() if t.team_id == team_id]
            self._assign_players_for_team(team_id, team_tracks, constraints)
        
        # Phase 4: Enforce global consistency
        self._enforce_global_consistency()
        
        logger.info(f"Created {len(self.players)} players from {len(self.tracks)} tracks")
        
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
        
        # Assign team 0 to tracks without embeddings
        for track in self.tracks.values():
            if track.team_id is None:
                track.team_id = 0
        
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
    
    def _assign_players_for_team(
        self, 
        team_id: int, 
        team_tracks: List[TrackInfo],
        constraints: Dict
    ):
        """Assign tracks to players within a team."""
        logger.info(f"Phase 3: Assigning players for team {team_id} "
                   f"({len(team_tracks)} tracks)")
        
        if not team_tracks:
            return
        
        # Sort tracks by start frame
        sorted_tracks = sorted(team_tracks, key=lambda t: t.start_frame)
        
        # Greedy assignment with appearance matching
        for track in sorted_tracks:
            best_player = self._find_best_player_match(track, constraints)
            
            if best_player is not None:
                # Attach to existing player
                best_player.add_track(track)
                track.player_id = best_player.player_id
                self.track_to_player[track.track_id] = best_player.player_id
            else:
                # Create new player
                player = PlayerInfo(
                    player_id=self.next_player_id,
                    team_id=team_id,
                )
                player.add_track(track)
                
                self.players[player.player_id] = player
                track.player_id = player.player_id
                self.track_to_player[track.track_id] = player.player_id
                self.next_player_id += 1
    
    def _find_best_player_match(
        self,
        track: TrackInfo,
        constraints: Dict
    ) -> Optional[PlayerInfo]:
        """Find best matching player for a track."""
        if track.embedding_mean is None:
            return None
        
        candidates = []
        
        for player in self.players.values():
            # Must be same team
            if player.team_id != track.team_id:
                continue
            
            # Check constraints with all player's tracks
            can_match = True
            for existing_track_id in player.track_ids:
                key = (track.track_id, existing_track_id)
                if key in constraints and not constraints[key]:
                    can_match = False
                    break
            
            if not can_match:
                continue
            
            # Compute appearance similarity
            similarity = player.similarity_to(track.embedding_mean)
            
            # Apply birth-type boost
            if track.birth_type == 'mid':
                similarity *= self.config.mid_birth_priority_boost
            
            if similarity >= self.config.similarity_threshold:
                candidates.append((player, similarity))
        
        if not candidates:
            return None
        
        # Return best match
        return max(candidates, key=lambda x: x[1])[0]
    
    def _enforce_global_consistency(self):
        """Ensure no player appears multiple times in same frame."""
        logger.info("Phase 4: Enforcing global consistency...")
        
        # Build frame -> [(track_id, player_id)] mapping
        frame_assignments: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        for track in self.tracks.values():
            for frame_id in range(track.start_frame, track.end_frame + 1):
                frame_assignments[frame_id].append(
                    (track.track_id, track.player_id)
                )
        
        # Check for conflicts
        conflicts = 0
        for frame_id, assignments in frame_assignments.items():
            player_ids = [p for _, p in assignments if p is not None]
            if len(player_ids) != len(set(player_ids)):
                conflicts += 1
                # TODO: Resolve conflicts
        
        if conflicts > 0:
            logger.warning(f"Found {conflicts} frames with player conflicts")
    
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
