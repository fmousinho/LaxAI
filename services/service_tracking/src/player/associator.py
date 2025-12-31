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
from collections import defaultdict
import json

from player.config import PlayerAssociatorConfig
from tracker.matching import linear_assignment

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
class Player:
    """Information about a discovered player."""
    player_id: int
    team_id: int
    state: Literal['active', 'lost', 'out_of_view'] = 'active'

    track_ids: List[int] = field(default_factory=list)
    track_segments: List[Tuple[int, int, int]] = field(default_factory=list)  # (track_id, start, end)
    embedding_bank: List[np.ndarray] = field(default_factory=list)

    lost_boundary: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    
    _matrix: Optional[np.ndarray] = None
    _dirty: bool = False
    
    def add_track(self, track: TrackInfo):
        """Add a track to this player."""
        self.track_ids.append(track.track_id)
        self.track_segments.append((track.track_id, track.start_frame, track.end_frame))
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
    
    def similarity_to(self, embeddings: np.ndarray, percent: float = .1) -> float:
        """Compute max similarity to any embedding in the bank."""
        if not self.embedding_bank or embeddings is None:
            return 0.0

        if self._dirty:
            self._matrix = np.vstack(self.embedding_bank)
            self._dirty = False
        
        sim_matrix = self._matrix @ embeddings.T
        flat = sim_matrix.ravel()

        k = max(1, int(len(flat) * percent))
        topk = np.partition(flat, -k)[-k:]
        return float(topk.mean())

    def to_dict(self) -> dict:
        return {
            'player_id': self.player_id,
            'team_id': self.team_id,
            'track_ids': self.track_ids,
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

        self.active_players: List[int] = []
        self.lost_players: List[int] = []
        self.out_of_view_players: List[int] = []

        # self.max_speed_px = (self.config.max_speed_meters_per_second * 
        #                self.config.pixels_per_meter / 
        #                self.config.fps)
        self.max_speed_px = 20
    
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
            track.birth_type = self._classify_birth(track)
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
        
        logger.info(f"Loaded {len(self.tracks)} tracks with {ttl_embeddings_count} embeddings")
    
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

        all_tracks = list(self.tracks.values())
        anchor_tracks = self._get_anchor_tracks(all_tracks, anchor_frame)
        logger.info(f"Anchor frame {anchor_frame} has {len(anchor_tracks)} tracks")

        for track in anchor_tracks:
            player_id, player = self._create_new_player(track)
            self.players[player_id] = player
            self.active_players.append(player_id)

        # Phase 3: Iterate forward and backwared from anchore frame and update player dictionary
        n_frames_processed = 0
        logger.info(f"Processing frames forward from anchor frame {anchor_frame}")  
        for frame_id in range(anchor_frame, self.total_frames):
            self._update_players_for_frame(frame_id, direction = "forward")
            n_frames_processed += 1
            if n_frames_processed % FRAMES_PER_SECOND == 0:
                logger.info(f"Processed {n_frames_processed}/{self.total_frames} frames, ttl players: {len(self.players)}")

        for frame_id in range(anchor_frame, -1, -1):
            self._update_players_for_frame(frame_id, direction = "backward")
            n_frames_processed += 1
            if n_frames_processed % FRAMES_PER_SECOND == 0:
                logger.info(f"Processed {n_frames_processed}/{self.total_frames} frames, ttl players: {len(self.players)}")

        
        logger.info(f"Final Count: {len(self.players)} players from {len(self.tracks)} tracks")
        
        return self.export()


    def _update_players_for_frame(self, frame_id: int, direction: Literal["forward", "backward"] = "forward"):
        """Update players for a given frame."""


        # 1. Get tracks for frame
        tids_in_frame = self.frame_tracks.get(frame_id, None)
        if not tids_in_frame:
            return

        logger.debug(f"===== Frame {frame_id} ======")

        tracks_in_frame = [self.tracks[tid] for tid in tids_in_frame]
        unassigned_tracks = [track for track in tracks_in_frame if not track.player_id]
        tracks_for_mid_matching = [track for track in unassigned_tracks if track.birth_type == 'mid']
        tracks_for_edge_matching = [track for track in unassigned_tracks if track.birth_type == 'edge']

        # 2. Get player ids in frame
        active_player_ids = [track.player_id for track in tracks_in_frame if track.player_id]

        lost_players = []
        lost_player_ids = []
        for player_id in self.active_players:
            if player_id not in active_player_ids:
                player = self.players[player_id]
                player.state = 'lost'
                if player.track_ids:
                    if direction == "forward":
                        last_track = self.tracks[player.track_ids[-1]]
                        bbox = last_track.last_bbox
                    else:
                        last_track = self.tracks[player.track_ids[0]]
                        bbox = last_track.first_bbox
                    player.lost_boundary = list(bbox)
                
                lost_players.append(player)
                lost_player_ids.append(player_id)

        logger.debug(
            f"Initially found {len(active_player_ids)} active players, lost {len(lost_player_ids)} players, "
            f"{len(unassigned_tracks)} unassigned tracks"
        )
    
        self._update_lost_boundary(lost_player_ids)
        self._update_lost_boundary(self.lost_players)

        for player_id in self.lost_players:
            player = self.players[player_id]
            if player.state == 'lost':
                lost_players.append(player)
            else:
                if player.player_id not in self.out_of_view_players:
                    self.out_of_view_players.append(player.player_id)
        
        current_lost_players = lost_players


        # 3. Try to find a match for orphan tracks in the middle of the frame (likely to be lost players)
        u_tracks_mid_objects = []
        if len(tracks_for_mid_matching) > 0 and len(current_lost_players) > 0:
            matches, u_tracks_ex, u_players_ex = self._match_tracks_to_players(
                tracks_for_mid_matching, current_lost_players, direction, threshold=1.0)
            
            # Match format from linear_assignment is usually np.ndarray of shape (N, 2) or list of tuples
            # Iterating assuming list of (track_idx, player_idx)
            for m in matches:
                track_idx, player_idx = m[0], m[1]
                track = tracks_for_mid_matching[track_idx]
                player = current_lost_players[player_idx]
                
                track.player_id = player.player_id
                player.add_track(track)
                player.state = 'active'
                self.track_to_player[track.track_id] = player.player_id
                active_player_ids.append(player.player_id)
                logger.debug(f"Matched mid track {track.track_id} to player {player.player_id}")
            
            # Isolate unmatched tracks objects
            for track_idx in u_tracks_ex:
                u_tracks_mid_objects.append(tracks_for_mid_matching[track_idx])
            
            # Update lost players - removing those that were matched
            # matched player indices are matches[:, 1]
            matched_player_indices = set(m[1] for m in matches)
            current_lost_players = [p for i, p in enumerate(current_lost_players) if i not in matched_player_indices]
            
        else:
            u_tracks_mid_objects = tracks_for_mid_matching

        # 4. Try to find a match for orphan tracks in the edge of the frame
        potential_edge_players = [p for p in current_lost_players] + [self.players[pid] for pid in self.out_of_view_players]
        u_tracks_edge_objects = []
        
        if len(tracks_for_edge_matching) > 0 and len(potential_edge_players) > 0:
            matches, u_tracks_ex, u_players_ex = self._match_tracks_to_players(
                tracks_for_edge_matching, potential_edge_players, direction, threshold=0.8)

            for m in matches:
                track_idx, player_idx = m[0], m[1]
                track = tracks_for_edge_matching[track_idx]
                player = potential_edge_players[player_idx]
                
                track.player_id = player.player_id
                player.add_track(track)
                player.state = 'active'
                self.track_to_player[track.track_id] = player.player_id
                active_player_ids.append(player.player_id)
                logger.debug(f"Matched edge track {track.track_id} to player {player.player_id}")
                
                if player.player_id in self.out_of_view_players:
                    self.out_of_view_players.remove(player.player_id)
            
            for track_idx in u_tracks_ex:
                u_tracks_edge_objects.append(tracks_for_edge_matching[track_idx])
        else:
            u_tracks_edge_objects = tracks_for_edge_matching
         
        # 5. Create new players for unassigned tracks
        unassigned_tracks_final = u_tracks_mid_objects + u_tracks_edge_objects
        for track in unassigned_tracks_final:
            if track.player_id is None and track.embeddings_count > 1:
                player_id, player = self._create_new_player(track)
                track.player_id = player_id
                active_player_ids.append(player_id)
                logger.debug(f"Created new player {player_id} from track {track.track_id}")

        self.active_players = list(set(active_player_ids))
        # Re-evaluate lost players from the global set
        self.lost_players = [pid for pid in self.players if self.players[pid].state == 'lost']

        self._log_players_states()

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
        direction: Literal["forward", "backward"] = "forward",
        threshold: float = 1.0
    ) -> Tuple[Dict[int, int], List[int], List[int]]:

        cost_matrix = np.zeros((len(tracks), len(players)))
        for i, track in enumerate(tracks):
            if direction == "forward":
                track_center = track.first_center
            else:
                track_center = track.last_center
            for j, player in enumerate(players):
                if self._track_is_in_player_lost_boundary(track_center, player):
                    cost_matrix[i, j] = player.similarity_to(track.embeddings_all)
                else:
                    cost_matrix[i, j] = np.inf

        return linear_assignment(cost_matrix, threshold)


    
    def _track_is_in_player_lost_boundary (self, track_center: Tuple[float, float], player: Player) -> bool:
        return (track_center[0] > player.lost_boundary[0] and \
            track_center[1] > player.lost_boundary[1] and \
            track_center[0] < player.lost_boundary[2] and \
            track_center[1] < player.lost_boundary[3]
            )
        
    
    
    def _create_new_player(self, track: TrackInfo) -> Tuple[int, Player]:
        player_id = len(self.players) + 1  # Start IDs from 1
        player = Player(player_id=player_id, team_id=track.team_id)
        player.add_track(track)
        track.player_id = player_id
        
        self.players[player_id] = player
        self.track_to_player[track.track_id] = player_id
        
        return player_id, player


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
            
            if (player.lost_boundary[0] < 0 or 
                player.lost_boundary[1] < 0 or 
                player.lost_boundary[2] > w or 
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
                track.team_id = 0
            return
            
        # 2. Prepare data for clustering
        X = np.concatenate(all_embeddings)
        logger.info(f"Clustering {len(X)} total embeddings from {len(tracks_with_data)} tracks")
        
        # 3. K-Means clustering
        kmeans = KMeans(n_clusters=self.config.n_teams, n_init=10, random_state=42)
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
            
            if self.config.verbose and confidence < 0.6:
                logger.debug(f"Track {tid}: Weak team consensus ({confidence:.2f}) {dict(votes)}")
                
            self.tracks[tid].team_id = best_team
            
    
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
    
    
    def _find_anchor_frame(self, tracks_by_frame: Dict[int, List[TrackInfo]]) -> int:
        """
        Find frame with tracks with the combined highest number of embeddings.
        """
        best_frame = None
        max_embeddings_count = 0

        for frame_id, tracks in tracks_by_frame.items():
            embeddings_count = 0
            for track in tracks:
                embeddings_count += track.embeddings_count if track.embeddings_count else 0
            if embeddings_count > max_embeddings_count:
                max_embeddings_count = embeddings_count
                best_frame = frame_id
        
        logger.info(f"Anchor frame: {best_frame} with {max_embeddings_count} embeddings")
        return best_frame
    
    def _get_anchor_tracks(self, tracks: List[TrackInfo], frame_id: int) -> List[TrackInfo]:
        """Get all tracks active in the anchor frame."""
        return [t for t in tracks if t.start_frame <= frame_id <= t.end_frame]
    
    
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
