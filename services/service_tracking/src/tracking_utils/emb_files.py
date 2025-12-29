import torch
import os
from typing import Dict, Any

def save_embeddings(embeddings: Dict[int, Dict[str, torch.Tensor]], path: str):
    """
    Saves the embeddings dictionary to a .pt file.
    
    Args:
        embeddings: Dictionary mapping track_id to a dictionary of tensors (e.g., {'mean': ..., 'variance': ...}).
        path: Destination path for the .pt file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(embeddings, path)

def load_embeddings(path: str) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Loads embeddings from a .pt file.
    
    Args:
        path: Path to the .pt file.
        
    Returns:
        Dictionary mapping track_id to embedding data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return torch.load(path)
