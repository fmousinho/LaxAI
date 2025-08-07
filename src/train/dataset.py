"""
Dataset utilities for the LaxAI project.

This module defines the LacrossePlayerDataset class and related utilities for loading and augmenting
lacrosse player image crops for training deep learning models, especially for triplet loss setups.
"""
import torch
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
    """
    def __init__(self, image_dir, storage_client=None, transform=None, min_images_per_player=training_config.min_images_per_player):
        self.image_dir = image_dir
        self.storage_client = storage_client
        self.transform = transform if transform is not None else get_transforms('training')
        self.min_images_per_player = min_images_per_player

        # Initialize dataset by loading player images
        if self.storage_client is None:
            raise ValueError("storage_client is required - local filesystem no longer supported")

        potential_players = self.storage_client.list_blobs(prefix=image_dir, delimiter='/')
        self.players = []
        self.player_to_images = {}

        # Group by identity (orig_crop_id) - the immediate subfolder under image_dir
        # For path structure: datasets/{dataset_id}/train/{orig_crop_id}/image.jpg
        player_to_images = {}
        for potential_player in potential_players:
            player_images = self.storage_client.list_blobs(prefix=potential_player)
            if len(player_images) > self.min_images_per_player:
                self.players.append(potential_player)
                self.player_to_images[potential_player] = player_images


        if len(self.players) < 2:
            raise ValueError(f"Need at least 2 players with {self.min_images_per_player}+ images each. Found {len(self.players)} valid players.")

        # Create list of all valid images
        self.all_images = []
        for player in self.players:
            self.all_images.extend(self.player_to_images[player])

        self.player_indices = {player: i for i, player in enumerate(self.players)}

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Dataset initialized with {len(self.players)} players and {len(self.all_images)} total images")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        # Anchor image
        anchor_blob = self.all_images[index]
        
        anchor_player = os.path.dirname(anchor_blob) + '/' 
        anchor_label = self.player_indices[anchor_player]

        try:
            anchor_img = self.storage_client.download_as_bytes(anchor_blob)
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
            positive_img = self.storage_client.download_as_bytes(positive_blob)
        except Exception as e:
            logger.error(f"Error loading positive image {positive_blob}: {e}")
            positive_img = anchor_img

        # Select a negative image (image of a different player)
        negative_candidates = [p for p in self.players if p != anchor_player]
        negative_player = random.choice(negative_candidates)
        negative_img_candidates = list(self.player_to_images[negative_player])
        negative_blob = random.choice(negative_img_candidates)
        try:
            negative_img = self.storage_client.download_as_bytes(negative_blob)
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




