import logging
logger = logging.getLogger(__name__)

from typing import Optional, Tuple

import torch
import torch.nn as nn

from shared_libs.config.all_config import model_config, training_config

from .model import LacrosseReIDResNet


class SiameseNet(nn.Module):
    """
    A Siamese network that uses a pre-trained ResNet with CBAM attention as a backbone
    to generate feature embeddings for player crops.
    
    This network is designed for player re-identification tasks in sports videos,
    producing normalized embeddings that can be used for similarity comparisons
    and clustering. The CBAM (Convolutional Block Attention Module) enhances
    feature representation through channel and spatial attention mechanisms.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize the SiameseNet with a CBAM-enhanced ResNet backbone.
        Accepts all optional parameters via kwargs for flexibility.
        
        Args:
            embedding_dim: Dimension of the output embedding vectors
            use_cbam: Whether to use CBAM attention modules
            attention_layers: List of layer names to apply CBAM to. 
                           Defaults to ['layer2', 'layer3', 'layer4']
            dropout_rate: Dropout rate for embedding layer
            Any other parameters can be passed via kwargs.
        """
        super().__init__()
        self.embedding_dim = kwargs.get('embedding_dim', model_config.embedding_dim)

        self.model = LacrosseReIDResNet(embedding_dim=self.embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.model.train()

        logger.info("=== Using Modified Resnet50 as backbone with CBAM ===")


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
        emb_anchor = self.model.forward(anchor)
        emb_positive = self.model.forward(positive)
        emb_negative = self.model.forward(negative)
        return emb_anchor, emb_positive, emb_negative
    
   
