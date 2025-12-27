"""
Configuration for Player identity management.
"""
from dataclasses import dataclass


@dataclass
class PlayerConfig:
    """Configuration for Player identity management system."""
    
    # Appearance matching
    similarity_threshold: float = 0.7
    """Minimum cosine similarity to match track to inactive player."""
    
    use_variance_weighting: bool = False
    """Use appearance variance for confidence weighting (experimental)."""
    
    # Temporal constraints
    max_inactive_gap: int = 150
    """Maximum frames between track end and new track to allow matching."""
    
    min_inactive_gap: int = 5
    """Minimum frames before allowing re-association (prevents flip-flopping)."""
    
    # Appearance model
    embedding_ema_alpha: float = 0.3
    """Exponential moving average weight for new embeddings (0-1)."""
    
    min_embeddings_for_match: int = 3
    """Minimum embeddings required before attempting to match to player."""
    
    # Player lifecycle
    auto_remove_after: int = 600
    """Remove inactive players after N frames to save memory (0 = never)."""
    
    # Logging
    verbose: bool = False
    """Enable detailed logging of matching decisions."""
