import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from core.config.all_config import model_config, training_config

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM."""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x


class ResNetWithCBAM(nn.Module):
    """ResNet backbone with CBAM attention modules."""
    
    def __init__(self, original_resnet, attention_layers=['layer2', 'layer3', 'layer4']):
        super(ResNetWithCBAM, self).__init__()
        
        # Copy all layers from original ResNet
        self.conv1 = original_resnet.conv1
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4
        
        self.avgpool = original_resnet.avgpool
        self.fc = original_resnet.fc
        
        # Add CBAM modules to specified layers
        self.attention_layers = attention_layers
        self.cbam_modules = nn.ModuleDict()
        
        if 'layer1' in attention_layers:
            self.cbam_modules['layer1'] = CBAM(64)  # ResNet18 layer1 output channels
        if 'layer2' in attention_layers:
            self.cbam_modules['layer2'] = CBAM(128)  # ResNet18 layer2 output channels
        if 'layer3' in attention_layers:
            self.cbam_modules['layer3'] = CBAM(256)  # ResNet18 layer3 output channels
        if 'layer4' in attention_layers:
            self.cbam_modules['layer4'] = CBAM(512)  # ResNet18 layer4 output channels
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'layer1' in self.attention_layers:
            x = self.cbam_modules['layer1'](x)
            
        x = self.layer2(x)
        if 'layer2' in self.attention_layers:
            x = self.cbam_modules['layer2'](x)
            
        x = self.layer3(x)
        if 'layer3' in self.attention_layers:
            x = self.cbam_modules['layer3'](x)
            
        x = self.layer4(x)
        if 'layer4' in self.attention_layers:
            x = self.cbam_modules['layer4'](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
        self.dropout_rate = kwargs.get('dropout_rate', model_config.dropout_rate)
        self.use_cbam = kwargs.get('use_cbam', True)
        attention_layers = kwargs.get('attention_layers', ['layer2', 'layer3', 'layer4'])

        # Create the original ResNet18
        original_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first conv layer to be more suitable for small images
        original_resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Get the number of features from the backbone's output
        num_ftrs = original_resnet.fc.in_features

        # Replace the final layer with our embedding layer including dropout
        original_resnet.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_ftrs, self.embedding_dim)
        )

        # Create backbone with or without CBAM
        if self.use_cbam:
            self.backbone = ResNetWithCBAM(original_resnet, attention_layers)
            logger.info(f"SiameseNet initialized with CBAM attention on layers: {attention_layers}")
        else:
            self.backbone = original_resnet
            logger.info(f"SiameseNet initialized without CBAM attention")

        logger.info(f"SiameseNet initialized with embedding dimension {self.embedding_dim}")
        logger.info(f"Using ResNet18 as backbone")
        logger.info(f"Dropout rate: {self.dropout_rate}")
        logger.info(f"First conv layer - kernel size: 3, stride: 1")

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
        embedding = F.normalize(embedding, p=2, dim=1)
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
    
    def get_attention_maps(self, x: torch.Tensor) -> dict:
        """
        Get attention maps from CBAM modules for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of attention maps from different layers
        """
        if not self.use_cbam:
            logger.warning("CBAM is not enabled. No attention maps available.")
            return {}
            
        attention_maps = {}
        
        # Forward pass through backbone to get intermediate attention maps
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if 'layer1' in self.backbone.attention_layers:
            channel_att = self.backbone.cbam_modules['layer1'].channel_attention(x)
            spatial_att = self.backbone.cbam_modules['layer1'].spatial_attention(x * channel_att)
            attention_maps['layer1'] = {'channel': channel_att, 'spatial': spatial_att}
            x = x * channel_att * spatial_att
            
        x = self.backbone.layer2(x)
        if 'layer2' in self.backbone.attention_layers:
            channel_att = self.backbone.cbam_modules['layer2'].channel_attention(x)
            spatial_att = self.backbone.cbam_modules['layer2'].spatial_attention(x * channel_att)
            attention_maps['layer2'] = {'channel': channel_att, 'spatial': spatial_att}
            x = x * channel_att * spatial_att
            
        x = self.backbone.layer3(x)
        if 'layer3' in self.backbone.attention_layers:
            channel_att = self.backbone.cbam_modules['layer3'].channel_attention(x)
            spatial_att = self.backbone.cbam_modules['layer3'].spatial_attention(x * channel_att)
            attention_maps['layer3'] = {'channel': channel_att, 'spatial': spatial_att}
            x = x * channel_att * spatial_att
            
        x = self.backbone.layer4(x)
        if 'layer4' in self.backbone.attention_layers:
            channel_att = self.backbone.cbam_modules['layer4'].channel_attention(x)
            spatial_att = self.backbone.cbam_modules['layer4'].spatial_attention(x * channel_att)
            attention_maps['layer4'] = {'channel': channel_att, 'spatial': spatial_att}
            x = x * channel_att * spatial_att
            
        return attention_maps
