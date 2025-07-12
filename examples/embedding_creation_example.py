# Example usage of the new create_embeddings_for_tracks function

"""
Example showing how to use the new create_embeddings_for_tracks method
with the AffineAwareByteTrack class.
"""

import torch
import numpy as np
from modules.custom_tracker import AffineAwareByteTrack

def example_embeddings_processor(crops_tensor: torch.Tensor) -> torch.Tensor:
    """
    Example embeddings processor function.
    In practice, this would be your actual model (e.g., SiameseNet, SigLip, etc.)
    
    Args:
        crops_tensor: Tensor of shape (batch_size, height, width, channels)
        
    Returns:
        Tensor of shape (batch_size, embedding_dim)
    """
    batch_size = crops_tensor.shape[0]
    embedding_dim = 512  # Example embedding dimension
    
    # Simulate model inference - replace with actual model call
    # Example: return model(crops_tensor)
    return torch.randn(batch_size, embedding_dim, dtype=torch.float32)

def example_usage():
    """Example of how to use the new embedding creation functionality."""
    
    # Initialize tracker
    tracker = AffineAwareByteTrack()
    
    # Assume we have some track data already populated from processing frames
    # (This would normally happen during video processing)
    
    # Create embeddings for all tracks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker.create_embeddings_for_tracks(
        embeddings_processor=example_embeddings_processor,
        device=device
    )
    
    # Get statistics about the embeddings
    stats = tracker.get_embedding_statistics()
    print("Embedding Statistics:")
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Tracks with embeddings: {stats['tracks_with_embeddings']}")
    print(f"  Total crops processed: {stats['total_crops']}")
    print(f"  Average crops per track: {stats.get('avg_crops_per_track', 0):.2f}")
    print(f"  Embedding dimensions: {stats['embedding_dimensions']}")
    
    # Access individual track embeddings
    for track_id, track_data in tracker.get_tracks_data().items():
        if track_data.embedding.size > 0:
            print(f"Track {track_id}: {track_data.num_crops} crops -> {track_data.embedding.shape} embedding")

if __name__ == "__main__":
    example_usage()
