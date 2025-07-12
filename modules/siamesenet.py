import logging
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from config.transforms_config import model_config, training_config

logger = logging.getLogger(__name__)


class SiameseNet(nn.Module):
    """
    A Siamese network that uses a pre-trained ResNet as a backbone
    to generate feature embeddings for player crops.
    
    This network is designed for player re-identification tasks in sports videos,
    producing normalized embeddings that can be used for similarity comparisons
    and clustering.
    """
    
    def __init__(self, embedding_dim: int = model_config.embedding_dim) -> None:
        """
        Initialize the SiameseNet with a ResNet backbone.
        
        Args:
            embedding_dim: Dimension of the output embedding vectors
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Use a pre-trained ResNet, but remove its final classification layer
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first conv layer to be more suitable for small images if needed
        # For example, smaller kernel and stride
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Get the number of features from the backbone's output
        num_ftrs = self.backbone.fc.in_features
        
        # Replace the final layer with our embedding layer
        self.backbone.fc = nn.Linear(num_ftrs, embedding_dim)

        logger.info(f"SiameseNet initialized with embedding dimension {embedding_dim}")
        logger.info(f"Using {self.backbone.__class__.__name__} as backbone")
        logger.info(f"First conv layer - kernel size: {self.backbone.conv1.kernel_size}, stride: {self.backbone.conv1.stride}")

        min_images_per_player=training_config.min_images_per_player

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate normalized embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Normalized embedding tensor of shape (batch_size, embedding_dim)
        """
        # The forward pass returns the embedding vector
        embedding = self.backbone(x)
        # L2-normalize the embedding
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def forward_triplet(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute embeddings for a triplet of inputs.
        
        Args:
            anchor: Anchor sample tensor
            positive: Positive sample tensor (similar to anchor)
            negative: Negative sample tensor (dissimilar to anchor)
            
        Returns:
            Tuple of (anchor_embedding, positive_embedding, negative_embedding)
        """
        emb_anchor = self.forward(anchor)
        emb_positive = self.forward(positive)
        emb_negative = self.forward(negative)
        return emb_anchor, emb_positive, emb_negative
