from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from typing import Optional

# Calculate root directory for .env location
# Path: services/service_tracking/src/player/offline_config.py -> src/player -> src -> service_tracking -> services -> root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
env_path = os.path.join(root_dir, ".env")


class PlayerAssociatorConfig(BaseSettings):
    """Configuration for offline (batch) player association."""

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Team clustering
    n_teams: int = Field(default=2, description="Number of teams to cluster (typically 2).")

    # Player discovery
    min_players_per_team: int = Field(default=11, description="Minimum players per team.")
    max_players_per_team: int = Field(
        default=22, description='Maximum players per team (including substitutes).'
    )
    default_players_per_team: int = Field(
        default=15, description='Default number of players per team if not auto-estimated.'
    )
    auto_estimate_player_count: bool = Field(
        default=True,
        description="Use silhouette analysis to estimate player count from anchors.",
    )

    # Spatial/velocity constraints
    fps: float = Field(default=30.0, description="Video frame rate.")
    max_speed_meters_per_second: float = Field(
        default=8.0, description="Max player speed in m/s (~22 mph, sprint speed)."
    )
    pixels_per_meter: float = Field(
        default=20.0,
        description="Approximate pixels per meter (will be estimated from frame).",
    )
    velocity_margin: float = Field(
        default=1.5,
        description="Safety margin for velocity feasibility (1.5 = allow 50% more than max).",
    )

    # Birth location heuristics
    field_top_ratio: float = Field(
        default=0.2,
        description="Top portion of frame that is NOT field (0.1 = top 10% is stands/bench).",
    )
    field_bottom_ratio: float = Field(
        default=1.0,
        description="Bottom portion of frame that IS field (1.0 = extends to bottom).",
    )
    edge_margin_ratio: float = Field(
        default=0.03,
        description='Portion of frame width/height considered "edge" (3% = ~60px on 1920 width).',
    )


    # Appearance matching
    similarity_threshold: float = Field(
        default=0.65, description="Minimum cosine similarity for track-to-player matching."
    )
    anchor_similarity_threshold: float = Field(
        default=0.7, description="Higher threshold for initial anchor clustering (more strict)."
    )
    max_embeddings_per_player: int = Field(
        default=50, description="Maximum embeddings to store per player (for embedding bank)."
    )
    use_embedding_bank: bool = Field(
        default=True,
        description="Use embedding bank instead of single mean (better for pose variance).",
    )

    # Output
    output_format: str = Field(default="players.json", description="Default output filename.")

    verbose: bool = Field(default=False, description="Enable detailed logging.")
