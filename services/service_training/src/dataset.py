"""
Dataset utilities for the LaxAI project.

This module defines the LacrossePlayerDataset class and related utilities for loading and augmenting
lacrosse player image crops for training deep learning models, especially for triplet loss setups.
"""
import logging
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from shared_libs.config.transforms import get_transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from shared_libs.config.all_config import training_config

logger = logging.getLogger(__name__)

class LacrossePlayerDataset(Dataset):
    """
    Custom Dataset for loading lacrosse player crops for triplet loss.
    Each player's crops are expected to be in a separate folder (prefix in GCS).
    
    Supports two modes:
    - Single dataset mode: Pass a single GCS blob prefix string (e.g., "datasets/dataset_123/train/")
    - Multi-dataset mode: Pass a list of GCS blob prefix strings (e.g., ["datasets/dataset_123/train/", "datasets/dataset_456/train/"])
    
    Args:
        image_dir: Either a single GCS blob prefix string or a list of GCS blob prefix strings.
                  Each string should point to a dataset directory containing player subdirectories.
                  Example: "tenant1/datasets/dataset_abc123/train/" or 
                          ["tenant1/datasets/dataset_abc123/train/", "tenant1/datasets/dataset_def456/train/"]
        storage_client: GoogleStorageClient instance for accessing GCS
        transform: Optional image transformations to apply
        min_images_per_player: Minimum number of images required per player to be included
    
    Behavior:
    - Single dataset mode: Negative samples are selected from any other player in the dataset
    - Multi-dataset mode: Negative samples are preferentially selected from other players within 
      the same dataset as the anchor. If no other players exist in the same dataset, falls back 
      to selecting from players in other datasets. This creates harder negatives by choosing 
      visually similar players from the same video/context.
    
    Expected GCS structure:
        tenant1/datasets/dataset_123/train/player_crop_id_1/image1.jpg
        tenant1/datasets/dataset_123/train/player_crop_id_1/image2.jpg
        tenant1/datasets/dataset_123/train/player_crop_id_2/image1.jpg
        tenant1/datasets/dataset_456/train/player_crop_id_3/image1.jpg
    """
    def __init__(self, image_dir: Union[str, List[str]], storage_client, transform=None, min_images_per_player=training_config.min_images_per_player, cache_size=1000):
        # Handle both single dataset and multi-dataset modes
        if isinstance(image_dir, str):
            self.dataset_list = [image_dir]
            self.multi_dataset_mode = False
        elif isinstance(image_dir, list):
            self.dataset_list = image_dir
            self.multi_dataset_mode = True
        else:
            raise ValueError("image_dir must be either a string (single dataset) or a list of strings (multi-dataset)")
            
        # Store tenant_id instead of the client object to avoid pickling issues
        # The client will be recreated when needed
        if storage_client is None:
            raise ValueError("storage_client is required - local filesystem no longer supported")
        
        self.storage_client = storage_client
        
        self.tenant_id = storage_client.user_id  # Store the tenant_id for recreating client
        self.transform = transform if transform is not None else get_transforms('training')
        self.min_images_per_player = min_images_per_player
        
        # Image caching to avoid repeated downloads
        self.cache_size = cache_size
        self._image_cache: Dict[str, Image.Image] = {}

        # Initialize dataset by loading player images using the provided client
        self.players = []
        self.player_to_images = {}
        self.dataset_to_players = {}

        # Process each dataset
        for dataset_dir in self.dataset_list:
            potential_players = storage_client.list_blobs(prefix=dataset_dir, delimiter='/')
            self.dataset_to_players[dataset_dir] = []
            for potential_player in potential_players:
                player_images = storage_client.list_blobs(prefix=potential_player)
                if len(player_images) > self.min_images_per_player:
                    self.players.append(potential_player)
                    self.player_to_images[potential_player] = player_images
                    self.dataset_to_players[dataset_dir].append(potential_player)

        if len(self.players) < 2:
            raise ValueError(f"Need at least 2 players with {self.min_images_per_player}+ images each. Found {len(self.players)} valid players.")

        # Create list of all valid images
        self.all_images = []
        for player in self.players:
            self.all_images.extend(self.player_to_images[player])

        self.player_indices = {player: i for i, player in enumerate(self.players)}
        self.dataset_indices = {dataset: i for i, dataset in enumerate(self.dataset_list)}

        # Create reverse mapping for optimization: player -> dataset (only needed in multi-dataset mode)
        if self.multi_dataset_mode:
            self.player_to_dataset = {}
            for dataset_dir, players in self.dataset_to_players.items():
                for player in players:
                    self.player_to_dataset[player] = dataset_dir

        # Pre-compute negative candidate lists for faster access
        self._precompute_negative_candidates()

        logger.info(f"Dataset initialized with {len(self.players)} players and {len(self.all_images)} total images")
        if self.multi_dataset_mode:
            logger.info(f"Multi-dataset mode: {len(self.dataset_list)} datasets")
        else:
            logger.info(f"Single dataset mode: {self.dataset_list[0]}")
        logger.info(f"Image cache size: {self.cache_size}")

    def _precompute_negative_candidates(self):
        """Pre-compute negative candidate lists for each player to avoid runtime computation."""
        self.negative_candidates = {}
        
        for player in self.players:
            if self.multi_dataset_mode:
                # Multi-dataset mode: prefer negatives from the same dataset
                player_dataset = self.player_to_dataset[player]
                same_dataset_candidates = [p for p in self.dataset_to_players[player_dataset] if p != player]
                
                # If no other players in the same dataset, fall back to any other player
                if not same_dataset_candidates:
                    candidates = [p for p in self.players if p != player]
                else:
                    candidates = same_dataset_candidates
            else:
                # Single dataset mode: select from any other player
                candidates = [p for p in self.players if p != player]
            
            self.negative_candidates[player] = candidates

    def _get_negative_candidates(self, anchor_player: str):
        """Get negative candidates for a player, computing lazily if needed."""
        # Ensure negative_candidates exists (handles unpickling)
        if not hasattr(self, 'negative_candidates'):
            self._precompute_negative_candidates()
        
        return self.negative_candidates[anchor_player]

    @lru_cache(maxsize=1000)
    def _get_cached_image(self, blob_path: str):
        """Cache frequently accessed images to avoid repeated downloads."""
        try:
            img = self.storage_client.download_as_appropriate_type(blob_path)
            
            # Check if download failed and returned None
            if img is None:
                logger.error(f"Failed to download image {blob_path} - returned None")
                return Image.new('RGB', (224, 224), color='black')
            
            # Convert numpy array to PIL Image if needed
            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                img = Image.fromarray(img)
                
            return img
        except Exception as e:
            logger.error(f"Error loading image {blob_path}: {e}")
            # Return a dummy image in case of error
            return Image.new('RGB', (224, 224), color='black')

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        # Anchor image
        anchor_blob = self.all_images[index]
        anchor_player = os.path.dirname(anchor_blob) + '/' 
        anchor_label = self.player_indices[anchor_player]

        # Use cached image loading
        anchor_img = self._get_cached_image(anchor_blob)

        # Select a positive image (different image of the same player)
        positive_list = self.player_to_images[anchor_player]
        if len(positive_list) < 2:
            positive_blob = anchor_blob
        else:
            positive_candidates = [p for p in positive_list if p != anchor_blob]
            positive_blob = random.choice(positive_candidates)
        
        positive_img = self._get_cached_image(positive_blob)

        # Select a negative image using pre-computed candidates
        negative_candidates = self._get_negative_candidates(anchor_player)
        negative_player = random.choice(negative_candidates)
        negative_img_candidates = self.player_to_images[negative_player]
        negative_blob = random.choice(list(negative_img_candidates))
        negative_img = self._get_cached_image(negative_blob)

        # Apply transformations - no need for type checking since _get_cached_image handles it
        if self.transform:
            try:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            except Exception as e:
                logger.error(f"Error applying transforms: {e}")
                anchor_img = transforms.ToTensor()(anchor_img)
                positive_img = transforms.ToTensor()(positive_img)
                negative_img = transforms.ToTensor()(negative_img)
        else:
            anchor_img = transforms.ToTensor()(anchor_img)
            positive_img = transforms.ToTensor()(positive_img)
            negative_img = transforms.ToTensor()(negative_img)

        return anchor_img, positive_img, negative_img, torch.tensor(anchor_label)




