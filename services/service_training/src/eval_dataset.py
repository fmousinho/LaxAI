import os
import io
from typing import Optional, Iterable

import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image


def load_eval_dataset(
    storage_client,
    image_paths: Iterable[str], 
    transform: Optional[transforms.Compose] = None
) -> TensorDataset:
    """Load evaluation dataset from image paths and create TensorDataset.
    
    Args:
        image_paths: Set of GCS blob paths to images
        transform: Optional image transformations to apply
        
    Returns:
        TensorDataset with (image_tensor, label_tensor) tuples
    """
    images_list = []
    labels_list = []
    player_to_idx = {}
    
    for blob_path in image_paths:
        # Extract player ID from path (player is the directory name)
        player = os.path.dirname(blob_path) + '/'
        
        # Assign player index
        if player not in player_to_idx:
            player_to_idx[player] = len(player_to_idx)
        label = player_to_idx[player]
        
        # Load and transform image
        img_bytes = storage_client.download_blob_to_bytes(blob_path)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if transform:
            img_tensor = transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        
        images_list.append(img_tensor)
        labels_list.append(label)
    
    # Stack into tensors
    images_tensor = torch.stack(images_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    
    # Create TensorDataset
    return TensorDataset(images_tensor, labels_tensor)