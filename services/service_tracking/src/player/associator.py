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
from typing import Dict, List, Optional, Tuple, Set, Literal
import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from collections import defaultdict, deque
import json

from player.config import PlayerAssociatorConfig
from tracker.matching import linear_assignment, enforce_min_distance

logger = logging.getLogger(__name__)

FRAMES_PER_SECOND = 100


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
    embeddings_all: Optional[np.ndarray] = None  # shape (embeddings_count, embedding_dim)
    embeddings_count: int = 0
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    
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
class Player:
    """Information about a discovered player."""
    player_id: int
    team_id: int
    state: Literal['active', 'lost', 'out_of_view'] = 'active'

    track_ids: deque[int] = field(default_factory=deque)
    track_segments: deque[Tuple[int, int, int]] = field(default_factory=deque)  # (track_id, start, end)
    embedding_bank: List[np.ndarray] = field(default_factory=list)

    lost_boundary: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    lost_frames: int = 0
    
    _matrix: Optional[np.ndarray] = None
    _dirty: bool = False
    
    def add_track(self, track: TrackInfo):
        """Add a track to this player."""
        if not self.track_segments or track.start_frame > self.track_segments[-1][2]:
            self.track_ids.append(track.track_id)
            self.track_segments.append((track.track_id, track.start_frame, track.end_frame))
        elif track.end_frame < self.track_segments[0][1]:
            self.track_ids.appendleft(track.track_id)
            self.track_segments.appendleft((track.track_id, track.start_frame, track.end_frame))
        else:
            raise ValueError(f"Track {track.track_id} is not in order with respect to player {self.player_id}")
        if track.embeddings_all is not None:
            # Add embeddings
            for emb in track.embeddings_all:
                self.embedding_bank.append(emb)
            
            # Simple cap implementation
            if len(self.embedding_bank) > 1000:
                # Keep recent 1000? Or random subsample?
                # Recent is probably safer for continuity but random is better for variety.
                # Let's keep last 1000 for now.
                self.embedding_bank = self.embedding_bank[-1000:]
                
            self._dirty = True
    
    def similarity_to(self, embeddings: np.ndarray, percent: float = 0.2) -> float:
        """
        Compute similarity to the embedding bank using cosine similarity.
        Returns the mean of the top-k most similar embeddings.
        """
        if not self.embedding_bank or embeddings is None:
            return 0.0

        if self._dirty:
            self._matrix = np.vstack(self.embedding_bank)
            self._dirty = False

        # Normalize (cosine similarity)
        bank = self._matrix / np.linalg.norm(self._matrix, axis=1, keepdims=True)
        emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Cosine similarity: (N_bank, N_query_embeddings)
        sims = bank @ emb.T
        
        # Flatten to get all pairwise similarities
        all_sims = sims.flatten()
        
        # Handle numerical issues
        all_sims = np.clip(all_sims, -1.0, 1.0)

        # Find top percent of all similarities
        k = max(1, int(len(all_sims) * percent))
        topk = np.partition(all_sims, -k)[-k:]

        return float(topk.mean())


    def to_dict(self) -> dict:
        return {
            'player_id': self.player_id,
            'team_id': self.team_id,
            'track_ids': list(self.track_ids),
            'track_segments': [[int(t), int(s), int(e)] for t, s, e in self.track_segments],
        }


class PlayerAssociator:
    """
    Associates tracks with players in an offline (batch) manner.
    """
    
    def __init__(self, config: PlayerAssociatorConfig):
        self.config = config
        self.tracks: Dict[int, TrackInfo] = {}
        self.players: Dict[int, PlayerInfo] = {}
        self.track_to_player: Dict[int, int] = {}
        self.next_player_id = 1
        self.frame_size: Tuple[int, int] = (1920, 1080)  # Default, will be set
        self.total_frames: int = 0
        self.frame_tracks: Dict[int, List[int]] = {}  # frame_id -> track_ids
        self.n_frames_processed = 0

        self.active_players: List[int] = []
        self.lost_players: List[int] = []
        self.out_of_view_players: List[int] = []
        self.direction: Literal["forward", "backward"] = "forward"

        # self.max_speed_px = (self.config.max_speed_meters_per_second * 
        #                self.config.pixels_per_meter / 
        #                self.config.fps)
        self.max_speed_px = 60
    
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

        frames_data = tracks_data.get('frames', [])

        self.total_frames = len(frames_data)
        
        for frame_data in frames_data:
            frame_id = frame_data['frame_id']
            self.frame_tracks[frame_id] = []
            for obj in frame_data.get('track_objects', []):
                track_id = obj['track_id']
                if track_id < 0:
                    continue  # Skip untracked detections
                
                bbox = tuple(obj.get('bbox', obj.get('tlbr', [0, 0, 0, 0])))
                track_frames[track_id].append((frame_id, bbox))
                self.frame_tracks[frame_id].append(track_id)
        
        # Create TrackInfo objects using classmethod
        for track_id, frames in track_frames.items():
            track = TrackInfo.from_frames_data(track_id, frames)
            self.tracks[track_id] = track
        
        # Load embeddings
        embeddings_data = torch.load(embeddings_path)
        ttl_embeddings_count = 0
        
        for track_id, emb_data in embeddings_data.items():
            if track_id in self.tracks:
                if isinstance(emb_data, dict):
                    if 'mean' in emb_data:
                        mean = emb_data['mean']
                        if isinstance(mean, torch.Tensor):
                            mean = mean.cpu().numpy()
                        mean = mean.flatten()
                        # Normalize mean
                        norm = np.linalg.norm(mean)
                        if norm > 0:
                            mean = mean / norm
                        self.tracks[track_id].embedding_mean = mean
                        
                    if 'variance' in emb_data:
                        var = emb_data['variance']
                        if isinstance(var, torch.Tensor):
                            var = var.cpu().numpy()
                        self.tracks[track_id].embedding_variance = var.flatten()
                        
                    if 'all' in emb_data:
                        embeddings = emb_data['all']
                        if isinstance(embeddings, torch.Tensor):
                            embeddings = embeddings.cpu().numpy()
                        
                        # Sanitize and normalize ALL embeddings
                        if embeddings is not None and len(embeddings) > 0:
                            # Handle NaNs
                            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # Normalize rows
                            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                            # Avoid division by zero
                            norms[norms == 0] = 1.0 
                            embeddings = embeddings / norms
                            
                            self.tracks[track_id].embeddings_all = embeddings
                    if 'count' in emb_data:
                        count = emb_data['count']
                        if isinstance(count, torch.Tensor):
                            count = count.cpu().numpy()
                        ttl_embeddings_count += count
                        self.tracks[track_id].embeddings_count = count
        
        # Summary statistics
        frames_without_tracks = sum(1 for fid in range(self.total_frames) if not self.frame_tracks.get(fid))
        tracks_without_embeddings = sum(1 for t in self.tracks.values() if t.embeddings_all is None)
        
        logger.info(f"Loaded {len(self.tracks)} tracks with {ttl_embeddings_count} embeddings")
        logger.info(f"Frames without tracks: {frames_without_tracks}/{self.total_frames}")
        logger.info(f"Tracks without embeddings: {tracks_without_embeddings}/{len(self.tracks)}")
    
    def run(self) -> Dict:
        """
        Run the full offline association pipeline.
        
        Returns:
            Dictionary with player assignments
        """
        logger.info("Starting offline player association...")
        
        # Phase 1: Identifies and assignes teams to tracks
        self._cluster_teams()
        n_tracks_team_0 = 0
        n_tracks_team_1 = 0
        for track in self.tracks.values():
            if track.team_id == 0:
                n_tracks_team_0 += 1
            elif track.team_id == 1:
                n_tracks_team_1 += 1
        logger.info(f"Found {n_tracks_team_0} tracks in team 0 and {n_tracks_team_1} tracks in team 1")

        # Phase 2: Finds best frame to use to initialize players
        # Build tracks by frame map for anchor search
        tracks_by_frame = {}
        for frame_id, tids in self.frame_tracks.items():
            tracks_by_frame[frame_id] = [self.tracks[tid] for tid in tids]
            
        anchor_frame = self._find_anchor_frame(tracks_by_frame)
        anchor_tracks = tracks_by_frame[anchor_frame]

        logger.info(f"Anchor frame {anchor_frame} has {len(anchor_tracks)} tracks")

        for track in anchor_tracks:
            player_id, player = self._create_new_player(track)
            if player_id is None:
                continue
            self.players[player_id] = player
            self.active_players.append(player_id) 

        # Phase 3: Iterate forward and backwared from anchore frame and update player dictionary
        logger.info(f"Processing frames forward from anchor frame {anchor_frame}")  
        for frame_id in range(anchor_frame, self.total_frames):
            self._update_players_for_frame(frame_id)
            self.n_frames_processed += 1
            if self.n_frames_processed % FRAMES_PER_SECOND == 0:
                logger.info(f"Processed {self.n_frames_processed}/{self.total_frames} frames, ttl players: {len(self.players)}")

        self.direction = "backward"
        for frame_id in range(anchor_frame, -1, -1):
            self._update_players_for_frame(frame_id)
            self.n_frames_processed += 1
            if self.n_frames_processed % FRAMES_PER_SECOND == 0:
                logger.info(f"Processed {self.n_frames_processed}/{self.total_frames} frames, ttl players: {len(self.players)}")


        # Count assigned tracks
        assigned_tracks = sum(1 for track in self.tracks.values() if track.player_id is not None)
        unassigned_tracks = len(self.tracks) - assigned_tracks
        
        logger.info(f"Final Count: {len(self.players)} players from {len(self.tracks)} tracks")
        logger.info(f"Assigned tracks: {assigned_tracks}/{len(self.tracks)} ({100*assigned_tracks/len(self.tracks):.1f}%)")
        logger.info(f"Unassigned tracks: {unassigned_tracks}")
        
        return self.export()


    def _update_players_for_frame(self, frame_id: int):
        """Update players for a given frame."""

        logger.debug(f"===== Frame {frame_id} ======")

        # 1. Get tracks for frame
        track_ids_in_frame = self.frame_tracks.get(frame_id, None)
        if not track_ids_in_frame:
            return

        tracks_in_frame = [self.tracks[tid] for tid in track_ids_in_frame]
        unassigned_tracks = [track for track in tracks_in_frame if not track.player_id]
            
        # 2. Creates updated lists of active, lost and oov players

        active_player_ids = [track.player_id for track in tracks_in_frame if track.player_id]
        lost_player_ids = []
        oov_player_ids = []

        for player_id, player in self.players.items():
            if player_id in active_player_ids:
                self._mark_active(player)
                continue
            if player.state == 'active':
                self._mark_lost(player)
                lost_player_ids.append(player_id)
                continue
            if player.state == 'lost':
                self._update_lost_player(player)
            if player.state == 'lost':  # players may lose lost status after the update, requiring a second pass
                lost_player_ids.append(player_id)
                continue
            if player.state == 'out_of_view':
                self._mark_out_of_view(player)
                oov_player_ids.append(player_id)
                continue
            raise ValueError(f"Player {player_id} is in an unknown state: {player.state} in frame {frame_id}")
            
        if len(self.players) > 0:
            logger.debug(
                f"Initially found {len(active_player_ids)} active players, lost {len(lost_player_ids)} players, "
                f"{len(unassigned_tracks)} unassigned tracks"
            )
            if len(unassigned_tracks) > 0:
                logger.debug("Unassigned tracks:")
                for track in unassigned_tracks:
                    logger.debug(f"Track {track.track_id} from frame {track.start_frame} to {track.end_frame}")

        # 3. Try to find a lost player for unmatched tracks
        if len(unassigned_tracks) > 0 and len(lost_player_ids) > 0:
            lost_players = [self.players[player_id] for player_id in lost_player_ids]
            matches, u_tracks_ex, u_players_ex = self._match_tracks_to_players(
                unassigned_tracks, lost_players, threshold=.1)
            
            for m in matches:
                track_idx, player_idx = m[0], m[1]
                track = unassigned_tracks[track_idx]
                player = lost_players[player_idx]
                self._associate(track, player)
                active_player_ids.append(player.player_id)
                logger.debug(f"Matched track {track.track_id} to LOST player {player.player_id}")
            
            unassigned_tracks = [unassigned_tracks[track_idx] for track_idx in u_tracks_ex]
            lost_player_ids = [lost_player_ids[player_idx] for player_idx in u_players_ex]

        # 4. Try to find a match for out of view players coming back to view
        unassigned_tracks_edge = []
        unassigned_tracks_middle = []
        for track in unassigned_tracks:
            center = track.first_center if self.direction == "forward" else track.last_center
            if self._classify_birth(center) == 'edge':
                unassigned_tracks_edge.append(track)
            else:
                unassigned_tracks_middle.append(track)

        if len(unassigned_tracks_edge) > 0 and len(oov_player_ids) > 0:
            oov_players = [self.players[player_id] for player_id in oov_player_ids]
            matches, u_tracks_ex, u_players_ex = self._match_tracks_to_players(
                unassigned_tracks_edge, oov_players, threshold=0.1)

            for m in matches:
                track_idx, player_idx = m[0], m[1]
                track = unassigned_tracks_edge[track_idx]
                player = oov_players[player_idx]
                self._associate(track, player)
                active_player_ids.append(player.player_id)
                logger.debug(f"Matched track {track.track_id} to OOV player {player.player_id}")
            
            unassigned_tracks_edge = [unassigned_tracks_edge[track_idx] for track_idx in u_tracks_ex]
            oov_player_ids = [oov_player_ids[player_idx] for player_idx in u_players_ex]   
         
        # 5. Create new players for unassigned tracks - do not create players for tracks that do not start in edge
        if self.n_frames_processed <= 300:
            tracks_for_new_players = unassigned_tracks_edge + unassigned_tracks_middle
        else:
            tracks_for_new_players = unassigned_tracks_edge
            if logger.getEffectiveLevel() == logging.DEBUG:
                for track in unassigned_tracks_middle:
                    track_center = track.first_center if self.direction == "forward" else track.last_center
                    logger.warning(f"No player will be created for Track {track.track_id}, center: {track_center}: not starting in edge")
        
        for track in tracks_for_new_players:
            if track.player_id is None and track.embeddings_count > 0:
                player_id, player = self._create_new_player(track)
                if player_id is None:
                    continue
                active_player_ids.append(player_id)
                logger.debug(f"Created new player {player_id} from track {track.track_id}")
            else:
                logger.debug(f"Track {track.track_id}, player id {track.player_id} has no embeddings, skipping")

        self.active_players = active_player_ids
        self.lost_players = lost_player_ids
        self.out_of_view_players = oov_player_ids

        self._log_players_states()

    def _mark_active(self, player: Player):
        player.state = 'active'
        player.lost_frames = 0

    def _mark_lost(self, player: Player):
        player.state = 'lost'
        player.lost_frames += 1
        margin_x = 100
        margin_y = 50
        if player.track_ids:
            # Get the correct track based on process direction
            tid = player.track_ids[-1] if self.direction == "forward" else player.track_ids[0]
            last_track = self.tracks[tid]
            
            # Use appropriate boundary based on direction
            raw_bbox = last_track.last_bbox if self.direction == "forward" else last_track.first_bbox
            # Convert to list to allow modification
            bbox = list(raw_bbox)
            bbox[0] -= margin_x
            bbox[1] -= margin_y
            bbox[2] += margin_x
            bbox[3] += margin_y
            player.lost_boundary = bbox

    def _mark_out_of_view(self, player: Player):
        player.state = 'out_of_view'

    def _associate(self, track: TrackInfo, player: Player):
        self._mark_active(player)
        track.player_id = player.player_id
        player.add_track(track)
        self.track_to_player[track.track_id] = player.player_id

    def _log_players_states(self):
        logger.debug("Active players:")
        logger.debug(self.active_players)
        logger.debug("Lost players:")
        for player_id in self.lost_players:
            lost_boundary_str = [int(x) for x in self.players[player_id].lost_boundary]
            logger.debug(f"Player {player_id}, lost boundary: {lost_boundary_str}")
        logger.debug("Out of view players:")
        logger.debug(self.out_of_view_players)
                
    def _match_tracks_to_players (
        self,
        tracks: List[TrackInfo],
        players: List[Player],
        threshold: float = 1.0
    ) -> Tuple[Dict[int, int], List[int], List[int]]:

        cost_matrix = np.zeros((len(tracks), len(players)))
        for i, track in enumerate(tracks):
            track_center = track.first_center if self.direction == "forward" else track.last_center
            logger.debug(f"Processing track {track.track_id}, center: {track_center}")
            for j, player in enumerate(players):
                # 0. Team ID check. Proceed if team track is unknown.
                if player.team_id != track.team_id and track.team_id != -1:
                    cost_matrix[i, j] = np.inf
                    continue

                # 1. Temporal Constraints: Check for overlap with existing player tracks
                has_overlap = False
                for seg in player.track_segments:
                    # seg is (tid, start, end)
                    # Check overlap: start1 <= end2 and start2 <= end1
                    if max(track.start_frame, seg[1]) <= min(track.end_frame, seg[2]):
                        has_overlap = True
                        logger.debug(f" - overlaps with player {player.player_id}")
                        break
                
                if has_overlap:
                    cost_matrix[i, j] = np.inf
                    continue

                # 2. Spatial Constraints
                if player.state == 'lost' and self._track_is_in_player_lost_boundary(track_center, player):
                    cost_matrix[i, j] = 1 - player.similarity_to(track.embeddings_all)
                    logger.debug(f" - matches lost player {player.player_id} with cost {cost_matrix[i, j]:.4f}")
                elif player.state == 'out_of_view':
                    cost_matrix[i, j] = 1 - player.similarity_to(track.embeddings_all)
                    logger.debug(f" - matches out of view player {player.player_id} with cost {cost_matrix[i, j]:.4f}")
                else:
                    cost_matrix[i, j] = np.inf
                    lost_boundary_str = [int(x) for x in player.lost_boundary]
                    logger.debug(f" - outside player {player.player_id} lost area {lost_boundary_str}")

        cost_matrix = enforce_min_distance(cost_matrix, min_distance = .01)

        return linear_assignment(cost_matrix, threshold)


    
    def _track_is_in_player_lost_boundary (self, track_center: Tuple[float, float], player: Player) -> bool:
        # If no boundary set, allow matching based on embeddings alone
        if player.lost_boundary is None:
            return True
            
        return (track_center[0] > player.lost_boundary[0] and \
            track_center[1] > player.lost_boundary[1] and \
            track_center[0] < player.lost_boundary[2] and \
            track_center[1] < player.lost_boundary[3]
            )
        
    
    
    def _get_last_player_center(self, player: Player) -> Optional[Tuple[float, float]]:
        """Get the center coordinates of the player's last known position."""
        if not player.track_ids:
            return None
        
        last_track_id = player.track_ids[-1]
        last_track = self.tracks[last_track_id]
        return last_track.last_center
    
    
    def _create_new_player(self, track: TrackInfo) -> Tuple[int, Player]:
        if track.team_id == -1:
            return None, None
        player_id = len(self.players) + 1  # Start IDs from 1
        player = Player(player_id=player_id, team_id=track.team_id)
        player.add_track(track)
        track.player_id = player_id
        self._mark_active(player)
        self.players[player_id] = player
        self.track_to_player[track.track_id] = player_id
        
        return player_id, player

    def _update_lost_player(self, player: Player):
        player.lost_frames += 1
        if player.lost_frames >=300:
            player.state = 'out_of_view'


    def _update_lost_boundary(self, player_ids: List[int]):
        w, h = self.frame_size
        
        for pid in player_ids:
            player = self.players[pid]
            # Calculate current center Y of the lost boundary for perspective scaling
            current_cy = (player.lost_boundary[1] + player.lost_boundary[3]) / 2
            
            # Perspective scaling: objects at the top (y=0) are smaller/further away.
            # We assume pixels_per_meter is calibrated for the near field (bottom).
            # Heuristic: Scale from 0.4 at top to 1.0 at bottom.
            perspective_scale = 0.4 + 0.6 * (current_cy / h)
            perspective_scale = max(0.2, min(1.0, perspective_scale))  # Clamp
            
            # Horizontal speed (X) scales with perspective
            speed_x = self.max_speed_px * perspective_scale
            
            # Vertical speed (Y) is further foreshortened by camera angle
            # (moving 1m into the field takes fewer pixels than 1m across)
            xy_aspect_ratio = 0.6  # Heuristic for typical sports camera angle
            speed_y = speed_x * xy_aspect_ratio
            
            player.lost_boundary[0] -= speed_x
            player.lost_boundary[1] -= speed_y
            player.lost_boundary[2] += speed_x
            player.lost_boundary[3] += speed_y
            
            if (player.lost_boundary[0] < -w or 
                player.lost_boundary[1] < 0 or 
                player.lost_boundary[2] > 2* w or 
                player.lost_boundary[3] > h):
                player.state = 'out_of_view'


    def _cluster_teams(self):
        """Cluster tracks into teams using K-Means on all available embeddings."""
        logger.info("Phase 1: Clustering tracks into teams (using all embeddings)...")
        
        all_embeddings = []
        embedding_track_map = []  # Index -> track_id
        tracks_with_data = set()
        
        # 1. Collect all embeddings
        for track in self.tracks.values():
            # Prefer comprehensive embeddings list
            if track.embeddings_all is not None and len(track.embeddings_all) > 0:
                all_embeddings.append(track.embeddings_all)
                embedding_track_map.extend([track.track_id] * len(track.embeddings_all))
                tracks_with_data.add(track.track_id)
            
            # Fallback to mean if that's all we have
            elif track.embedding_mean is not None:
                all_embeddings.append(track.embedding_mean.reshape(1, -1))
                embedding_track_map.append(track.track_id)
                tracks_with_data.add(track.track_id)
        
        if len(tracks_with_data) < 2:
            logger.warning("Not enough tracks with embeddings for team clustering")
            for track in self.tracks.values():
                track.team_id = -1
            return
            
        # 2. Prepare data for clustering
        X = np.concatenate(all_embeddings)
        logger.info(f"Clustering {len(X)} total embeddings from {len(tracks_with_data)} tracks")
        
        # 3. K-Means clustering
        kmeans = KMeans(n_clusters=self.config.n_teams, n_init=8, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # 4. Voting mechanism
        track_votes = defaultdict(lambda: defaultdict(int))
        for i, label in enumerate(labels):
            tid = embedding_track_map[i]
            track_votes[tid][int(label)] += 1
            
        # 5. Assign teams based on majority vote
        for tid, votes in track_votes.items():
            # Get key with max value
            best_team = max(votes.items(), key=lambda x: x[1])[0]
            
            # Calculate confidence (for logging/debugging could be useful)
            total_votes = sum(votes.values())
            confidence = votes[best_team] / total_votes
            
            if confidence < self.config.min_voting_confidence_for_team_assignment:
                logger.debug(f"Track {tid}: Weak team consensus ({confidence:.2f}) {dict(votes)}")
                self.tracks[tid].team_id = -1
            else:
                self.tracks[tid].team_id = best_team
            
    
    def _classify_birth(self, position: Tuple[int, int]) -> str:
        """Classify track birth as 'edge' or 'mid'."""
        w, h = self.frame_size
        x, y = position
        
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
    
    
    def _find_anchor_frame(self, tracks_by_frame: Dict[int, List[TrackInfo]]) -> int:
        """
        Find frame with tracks with the combined highest number of embeddings.
        """
        best_frame = None
        max_embeddings_count = 0

        upper_margin = self.frame_size[1]

        for frame_id, tracks in tracks_by_frame.items():
            embeddings_count = 0
            for track in tracks:
                embeddings_count += track.embeddings_count if track.embeddings_count else 0
                if track.first_bbox[1] < upper_margin:
                    upper_margin = track.first_bbox[1]
            if embeddings_count > max_embeddings_count:
                max_embeddings_count = embeddings_count
                best_frame = frame_id
                
        margin_buffer = 100  # pixels
        self.config.field_top_ratio = (upper_margin + margin_buffer) / self.frame_size[1]
        
        logger.info(f"Anchor frame: {best_frame} with {max_embeddings_count} embeddings")
        return best_frame
    
    
    
    def export(self) -> Dict:
        """Export results to dictionary."""
        teams = defaultdict(list)
        track_to_player_map = {}
        
        for pid, player in self.players.items():
            teams[player.team_id].append(pid)
            for tid in player.track_ids:
                track_to_player_map[str(tid)] = {
                    'player_id': pid, 
                    'team_id': player.team_id
                }
        
        return {
            'teams': {
                str(team_id): {'player_ids': pids}
                for team_id, pids in teams.items()
            },
            'players': {
                str(pid): player.to_dict()
                for pid, player in self.players.items()
            },
            'track_to_player': track_to_player_map,
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
