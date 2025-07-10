# Example usage of the embedding creation functions

"""
Example showing how to use the new embedding creation functions
in train_processor.py
"""

import torch
import numpy as np
import cv2
from modules.train_processor import create_embeddings_from_crops, create_averaged_embedding_from_crops
from modules.siamesenet import SiameseNet  # Example model

def load_example_crops(num_crops: int = 10) -> list:
    """
    Create example crops for demonstration.
    In practice, these would be loaded from your crop extraction process.
    """
    crops = []
    for i in range(num_crops):
        # Create random crop images (in practice, these would be real crops)
        crop = np.random.randint(0, 255, (64, 32, 3), dtype=np.uint8)  # Example: 64x32 RGB crop
        crops.append(crop)
    return crops

def example_batch_embedding_creation():
    """Example of creating embeddings from crops using batch processing."""
    
    # Load a trained model
    model = SiameseNet(embedding_dim=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Example: Load model weights if available
    # model.load_state_dict(torch.load("path/to/model.pth"))
    
    # Get example crops
    crops = load_example_crops(num_crops=50)
    print(f"Processing {len(crops)} crops...")
    
    # Method 1: Create individual embeddings for each crop
    all_embeddings = create_embeddings_from_crops(
        crops=crops,
        model=model,
        batch_size=16,  # Process 16 crops at a time
        device=device
    )
    
    print(f"Created {all_embeddings.shape[0]} individual embeddings")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    # Method 2: Create a single averaged embedding from all crops
    averaged_embedding = create_averaged_embedding_from_crops(
        crops=crops,
        model=model,
        batch_size=16,
        device=device
    )
    
    print(f"Created averaged embedding with dimension: {averaged_embedding.shape[0]}")
    
    return all_embeddings, averaged_embedding

def example_track_embeddings():
    """Example of creating embeddings for multiple tracks."""
    
    # Simulate track data (in practice, this would come from your tracker)
    track_crops = {
        1: load_example_crops(15),  # Track 1 has 15 crops
        2: load_example_crops(8),   # Track 2 has 8 crops
        3: load_example_crops(23),  # Track 3 has 23 crops
    }
    
    # Load model
    model = SiameseNet(embedding_dim=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    track_embeddings = {}
    
    # Create averaged embedding for each track
    for track_id, crops in track_crops.items():
        print(f"Processing track {track_id} with {len(crops)} crops...")
        
        # Create averaged embedding for this track
        track_embedding = create_averaged_embedding_from_crops(
            crops=crops,
            model=model,
            batch_size=16,
            device=device
        )
        
        track_embeddings[track_id] = track_embedding
        print(f"Track {track_id}: {track_embedding.shape} embedding created")
    
    return track_embeddings

def example_similarity_calculation():
    """Example of calculating similarities between track embeddings."""
    
    # Get track embeddings
    track_embeddings = example_track_embeddings()
    
    # Calculate cosine similarities between tracks
    from sklearn.metrics.pairwise import cosine_similarity
    
    track_ids = list(track_embeddings.keys())
    embeddings_matrix = np.stack([track_embeddings[tid] for tid in track_ids])
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    print("\nTrack Similarity Matrix:")
    print("Track IDs:", track_ids)
    for i, tid1 in enumerate(track_ids):
        for j, tid2 in enumerate(track_ids):
            print(f"Track {tid1} vs Track {tid2}: {similarity_matrix[i, j]:.3f}")

if __name__ == "__main__":
    print("=== Batch Embedding Creation Example ===")
    example_batch_embedding_creation()
    
    print("\n=== Track Embeddings Example ===")
    example_track_embeddings()
    
    print("\n=== Similarity Calculation Example ===")
    example_similarity_calculation()
