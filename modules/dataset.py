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
    Each player's crops are expected to be in a separate folder.
    """
    def __init__(self, image_dir, transform=None, min_images_per_player=training_config.min_images_per_player):
        self.image_dir = image_dir
        self.transform = transform
        self.min_images_per_player = min_images_per_player
        
        # Get all player directories
        all_players = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        
        # Filter players with sufficient images
        self.players = []
        self.player_to_images = {}
        
        for player in all_players:
            player_images = [os.path.join(image_dir, player, img) 
                           for img in os.listdir(os.path.join(image_dir, player))
                           if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            # Only include players with enough images for triplet sampling
            if len(player_images) >= self.min_images_per_player:
                self.players.append(player)
                self.player_to_images[player] = player_images
        
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
        anchor_path = self.all_images[index]
        anchor_player = os.path.basename(os.path.dirname(anchor_path))
        anchor_label = self.player_indices[anchor_player]
        
        try:
            anchor_img = Image.open(anchor_path).convert('RGB')
        except Exception as e:
            print(f"Error loading anchor image {anchor_path}: {e}")
            # Return the first valid image as fallback
            anchor_path = self.all_images[0]
            anchor_player = os.path.basename(os.path.dirname(anchor_path))
            anchor_label = self.player_indices[anchor_player]
            anchor_img = Image.open(anchor_path).convert('RGB')

        # Select a positive image (different image of the same player)
        positive_list = self.player_to_images[anchor_player]
        if len(positive_list) < 2:
            # If only one image, use the same image (shouldn't happen due to filtering)
            positive_path = anchor_path
        else:
            # Ensure positive is different from anchor
            positive_candidates = [p for p in positive_list if p != anchor_path]
            if positive_candidates:
                positive_path = random.choice(positive_candidates)
            else:
                positive_path = random.choice(positive_list)  # Fallback
        
        try:
            positive_img = Image.open(positive_path).convert('RGB')
        except Exception as e:
            print(f"Error loading positive image {positive_path}: {e}")
            positive_img = anchor_img  # Use anchor as fallback
        
        # Select a negative image (image of a different player)
        negative_candidates = [p for p in self.players if p != anchor_player]
        if not negative_candidates:
            # This shouldn't happen if we have at least 2 players
            negative_player = anchor_player
        else:
            negative_player = random.choice(negative_candidates)
        
        negative_path = random.choice(self.player_to_images[negative_player])
        
        try:
            negative_img = Image.open(negative_path).convert('RGB')
        except Exception as e:
            print(f"Error loading negative image {negative_path}: {e}")
            negative_img = anchor_img  # Use anchor as fallback

        # Apply transformations
        if self.transform:
            try:
                # Always convert PIL Images to numpy arrays for transforms
                # This works with both regular and OpenCV-safe transforms
                anchor_array = np.array(anchor_img)
                positive_array = np.array(positive_img) 
                negative_array = np.array(negative_img)
                
                anchor_img = self.transform(anchor_array)
                positive_img = self.transform(positive_array)
                negative_img = self.transform(negative_array)
            except Exception as e:
                print(f"Error applying transforms: {e}")
                print(f"Transform type: {type(self.transform)}")
                print(f"Transform: {self.transform}")
                # Return tensors without transforms as fallback
                anchor_img = transforms.ToTensor()(anchor_img)
                positive_img = transforms.ToTensor()(positive_img)
                negative_img = transforms.ToTensor()(negative_img)
        else:
            # If no transforms provided, at minimum convert to tensor
            anchor_img = transforms.ToTensor()(anchor_img)
            positive_img = transforms.ToTensor()(positive_img)
            negative_img = transforms.ToTensor()(negative_img)

        return anchor_img, positive_img, negative_img, torch.tensor(anchor_label)

# Get default training transforms from centralized config
data_transforms = get_transforms('training')
