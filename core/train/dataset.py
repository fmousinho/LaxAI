"""
Dataset utilities for the LaxAI project.

This module defines the LacrossePlayerDataset class and related utilities for loading and augmenting
lacrosse player image crops for training deep learning models, especially for triplet loss setups.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np
from config.transforms import get_transforms

from config.all_config import training_config

class LacrossePlayerDataset(Dataset):
    """
    Custom Dataset for loading lacrosse player crops for triplet loss.
    Each player's crops are expected to be in a separate folder (prefix in GCS).
    """
    def __init__(self, image_dir, storage_client, transform=None, min_images_per_player=training_config.min_images_per_player):
        self.image_dir = image_dir
        self.storage_client = storage_client
        self.transform = transform if transform is not None else get_ad('training')
        self.min_images_per_player = min_images_per_player

        # List all blobs under image_dir
        all_blobs = self.storage_client.list_blobs(prefix=image_dir)
        # Filter for image files
        image_blobs = [b for b in all_blobs if b.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Group by player (assume player is the immediate subfolder under image_dir)
        player_to_images = {}
        for blob in image_blobs:
            parts = blob[len(image_dir):].lstrip('/').split('/')
            if len(parts) < 2:
                continue  # Not in a player subfolder
            player = parts[0]
            player_to_images.setdefault(player, []).append(blob)

        # Filter players with sufficient images
        self.players = []
        self.player_to_images = {}
        for player, imgs in player_to_images.items():
            if len(imgs) >= self.min_images_per_player:
                self.players.append(player)
                self.player_to_images[player] = imgs

        if len(self.players) < 2:
            raise ValueError(f"Need at least 2 players with {min_images_per_player}+ images each. Found {len(self.players)} valid players.")

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
        anchor_player = anchor_blob[len(self.image_dir):].lstrip('/').split('/')[0]
        anchor_label = self.player_indices[anchor_player]

        try:
            anchor_img = self._load_image_from_gcs(anchor_blob)
        except Exception as e:
            print(f"Error loading anchor image {anchor_blob}: {e}")
            anchor_blob = self.all_images[0]
            anchor_player = anchor_blob[len(self.image_dir):].lstrip('/').split('/')[0]
            anchor_label = self.player_indices[anchor_player]
            anchor_img = self._load_image_from_gcs(anchor_blob)

        # Select a positive image (different image of the same player)
        positive_list = self.player_to_images[anchor_player]
        if len(positive_list) < 2:
            positive_blob = anchor_blob
        else:
            positive_candidates = [p for p in positive_list if p != anchor_blob]
            if positive_candidates:
                positive_blob = random.choice(positive_candidates)
            else:
                positive_blob = random.choice(positive_list)
        try:
            positive_img = self._load_image_from_gcs(positive_blob)
        except Exception as e:
            print(f"Error loading positive image {positive_blob}: {e}")
            positive_img = anchor_img

        # Select a negative image (image of a different player)
        negative_candidates = [p for p in self.players if p != anchor_player]
        if not negative_candidates:
            negative_player = anchor_player
        else:
            negative_player = random.choice(negative_candidates)
        negative_blob = random.choice(self.player_to_images[negative_player])
        try:
            negative_img = self._load_image_from_gcs(negative_blob)
        except Exception as e:
            print(f"Error loading negative image {negative_blob}: {e}")
            negative_img = anchor_img

        # Apply transformations
        if self.transform:
            try:
                anchor_array = np.array(anchor_img)
                positive_array = np.array(positive_img)
                negative_array = np.array(negative_img)
                anchor_img = self.transform(anchor_array)
                positive_img = self.transform(positive_array)
                negative_img = self.transform(negative_array)
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

    def _load_image_from_gcs(self, blob_path):
        # Download blob to a temporary file and open with PIL
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(blob_path)[-1]) as tmp:
            if not self.storage_client.download_blob(blob_path, tmp.name):
                raise IOError(f"Failed to download blob: {blob_path}")
            return Image.open(tmp.name).convert('RGB')


