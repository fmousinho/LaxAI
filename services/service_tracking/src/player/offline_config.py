"""
Configuration for offline player association.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class OfflinePlayerConfig:
    """Configuration for offline (batch) player association."""
    
    # Team clustering
    n_teams: int = 2
    """Number of teams to cluster (typically 2)."""
    
    # Player discovery
    min_players_per_team: int = 11
    """Minimum players per team."""
    
    max_players_per_team: int = 22  
    """Maximum players per team (including substitutes)."""
    
    default_players_per_team: int = 15
    """Default number of players per team if not auto-estimated."""
    
    auto_estimate_player_count: bool = True
    """Use silhouette analysis to estimate player count from anchors."""
    
    # Spatial/velocity constraints
    fps: float = 30.0
    """Video frame rate."""
    
    max_speed_meters_per_second: float = 10.0
    """Max player speed in m/s (~22 mph, sprint speed)."""
    
    pixels_per_meter: float = 20.0
    """Approximate pixels per meter (will be estimated from frame)."""
    
    velocity_margin: float = 1.5
    """Safety margin for velocity feasibility (1.5 = allow 50% more than max)."""
    
    # Birth location heuristics
    field_top_ratio: float = 0.1
    """Top portion of frame that is NOT field (0.1 = top 10% is stands/bench)."""
    
    field_bottom_ratio: float = 1.0
    """Bottom portion of frame that IS field (1.0 = extends to bottom)."""
    
    edge_margin_ratio: float = 0.03
    """Portion of frame width/height considered "edge" (3% = ~60px on 1920 width)."""
    
    mid_birth_priority_boost: float = 1.5
    """Similarity boost for mid-frame births (more likely to be lost player)."""
    
    # Appearance matching
    similarity_threshold: float = 0.65
    """Minimum cosine similarity for track-to-player matching."""
    
    anchor_similarity_threshold: float = 0.70
    """Higher threshold for initial anchor clustering (more strict)."""
    
    max_embeddings_per_player: int = 50
    """Maximum embeddings to store per player (for embedding bank)."""
    
    use_embedding_bank: bool = True
    """Use embedding bank instead of single mean (better for pose variance)."""
    
    # Output
    output_format: str = "players.json"
    """Default output filename."""
    
    verbose: bool = False
    """Enable detailed logging."""
