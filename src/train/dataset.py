"""
Dataset utilities for the LaxAI project.

This module defines the LacrossePlayerDataset class and related utilities for loading and augmenting
lacrosse player image crops for training deep learning models, especially for triplet loss setups.
"""
import torch
from typing import List, Union
import logging
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np
from config.transforms import get_transforms

from config.all_config import training_config

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
    def __init__(self, image_dir: Union[str, List[str]], storage_client, transform=None, min_images_per_player=training_config.min_images_per_player):
        # Handle both single dataset and multi-dataset modes
        if isinstance(image_dir, str):
            self.dataset_list = [image_dir]
            self.multi_dataset_mode = False
        elif isinstance(image_dir, list):
            self.dataset_list = image_dir
            self.multi_dataset_mode = True
        else:
            raise ValueError("image_dir must be either a string (single dataset) or a list of strings (multi-dataset)")
            
        self.storage_client = storage_client
        self.transform = transform if transform is not None else get_transforms('training')
        self.min_images_per_player = min_images_per_player

        # Initialize dataset by loading player images
        if self.storage_client is None:
            raise ValueError("storage_client is required - local filesystem no longer supported")

        self.players = []
        self.player_to_images = {}
        self.dataset_to_players = {}

        # Process each dataset
        for dataset_dir in self.dataset_list:
            potential_players = self.storage_client.list_blobs(prefix=dataset_dir, delimiter='/')
            self.dataset_to_players[dataset_dir] = []
            for potential_player in potential_players:
                player_images = self.storage_client.list_blobs(prefix=potential_player)
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

        logger.info(f"Dataset initialized with {len(self.players)} players and {len(self.all_images)} total images")
        if self.multi_dataset_mode:
            logger.info(f"Multi-dataset mode: {len(self.dataset_list)} datasets")
        else:
            logger.info(f"Single dataset mode: {self.dataset_list[0]}")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        # Anchor image
        anchor_blob = self.all_images[index]
        
        anchor_player = os.path.dirname(anchor_blob) + '/' 
        anchor_label = self.player_indices[anchor_player]

        try:
            anchor_img = self.storage_client.download_as_appropriate_type(anchor_blob)
        except Exception as e:
            logger.error(f"Error loading anchor image {anchor_blob}: {e}")

        # Select a positive image (different image of the same player)
        positive_list = self.player_to_images[anchor_player]
        if len(positive_list) < 2:
            positive_blob = anchor_blob
        else:
            positive_candidates = [p for p in positive_list if p != anchor_blob]
            positive_blob = random.choice(positive_candidates)
        
        try:
            positive_img = self.storage_client.download_as_appropriate_type(positive_blob)
        except Exception as e:
            logger.error(f"Error loading positive image {positive_blob}: {e}")
            positive_img = anchor_img

        # Select a negative image (different player, preferring same dataset in multi-dataset mode)
        if self.multi_dataset_mode:
            # Multi-dataset mode: prefer negatives from the same dataset
            anchor_dataset = self.player_to_dataset[anchor_player]
            negative_candidates = [p for p in self.dataset_to_players[anchor_dataset] if p != anchor_player]
            
            # If no other players in the same dataset, fall back to any other player
            if not negative_candidates:
                logger.warning(f"No other players in dataset {anchor_dataset}, using player from different dataset")
                negative_candidates = [p for p in self.players if p != anchor_player]
        else:
            # Single dataset mode: select from any other player
            negative_candidates = [p for p in self.players if p != anchor_player]
            
        negative_player = random.choice(negative_candidates)
        negative_img_candidates = list(self.player_to_images[negative_player])
        negative_blob = random.choice(negative_img_candidates)
        try:
            negative_img = self.storage_client.download_as_appropriate_type(negative_blob)
        except Exception as e:
            logger.error(f"Error loading negative image {negative_blob}: {e}")
            negative_img = anchor_img

        # Apply transformations

        if isinstance(anchor_img, np.ndarray) and anchor_img.ndim == 3 and anchor_img.shape[2] == 3:
            anchor_img = Image.fromarray(anchor_img)
        if isinstance(positive_img, np.ndarray) and positive_img.ndim == 3 and positive_img.shape[2] == 3:
            positive_img = Image.fromarray(positive_img)
        if isinstance(negative_img, np.ndarray) and negative_img.ndim == 3 and negative_img.shape[2] == 3:
            negative_img = Image.fromarray(negative_img)
       

        if self.transform:
            try:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            except Exception as e:
                print(f"Error applying transforms: {e}")
                anchor_img = transforms.ToTensor()(anchor_img)
                positive_img = transforms.ToTensor()(positive_img)
                negative_img = transforms.ToTensor()(negative_img)
        else:
            anchor_img = transforms.ToTensor()(anchor_img)
            positive_img = transforms.ToTensor()(positive_img)
            negative_img = transforms.ToTensor()(negative_img)

        return anchor_img, positive_img, negative_img, torch.tensor(anchor_label)




