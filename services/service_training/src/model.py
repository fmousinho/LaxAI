import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

# ==========================================
# 1. The CBAM Module (Channel + Spatial)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ==========================================
# 2. The Modified Bottleneck (With CBAM)
# ==========================================
class BottleneckCBAM(nn.Module):
    # This mimics the standard ResNet Bottleneck but adds CBAM before the residual add
    expansion = 4

    def __init__(self, original_block, cbam_ratio=16):
        super(BottleneckCBAM, self).__init__()
        # Copy layers from the original pretrained block
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.conv3 = original_block.conv3
        self.bn3 = original_block.bn3
        self.relu = original_block.relu
        self.downsample = original_block.downsample
        self.stride = original_block.stride

        # Add CBAM layer (Input channels to CBAM is output of conv3)
        self.cbam = CBAM(original_block.conv3.out_channels, ratio=cbam_ratio)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # --- CBAM INSERTION ---
        # Apply attention maps to the features before adding the identity
        out = self.cbam(out)
        # ----------------------

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# ==========================================
# 3. The Main Network Class
# ==========================================
class LacrosseReIDResNet(nn.Module):
    model_name = "LacrosseReIDNet - ResNet50 with CBAM and BNNeck v1.0"

    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        
        # 1. Load Standard Pretrained Weights
        if pretrained:
            logger.info("Loading ResNet50 ImageNet weights...")
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            logger.info("Skipping ResNet50 ImageNet weights (pretrained=False)...")
            self.base_model = resnet50(weights=None)
        
        # 2. Apply the Stride Trick (Layer 4 stride = 1)
        # This increases feature map size from 8x4 to 16x8 (for 256x128 input)
        self.base_model.layer4[0].conv2.stride = (1, 1)
        self.base_model.layer4[0].downsample[0].stride = (1, 1)
        
        # 3. Inject CBAM into Layers 2, 3, and 4
        # We replace the standard Bottleneck with our BottleneckCBAM
        # We do this AFTER loading weights so we inherit the conv weights
        self._replace_with_cbam(self.base_model.layer2)
        self._replace_with_cbam(self.base_model.layer3)
        self._replace_with_cbam(self.base_model.layer4)

        # 4. Define the BNNeck Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False) 
        self.classifier = nn.Linear(2048, embedding_dim, bias=False)
        
        # Init head weights
        self.bottleneck.apply(self._weights_init_kaiming)
        self.classifier.apply(self._weights_init_classifier)

    def _replace_with_cbam(self, layer):
        """
        Iterates through a ResNet layer (Sequential) and replaces 
        Standard Bottlenecks with CBAM Bottlenecks, keeping weights.
        """
        for i, block in enumerate(layer):
            # Create new block wrapping the old one
            layer[i] = BottleneckCBAM(block)

    def forward(self, x):
        # Standard ResNet forward pass extracted from base_model
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x) # Now contains CBAM
        x = self.base_model.layer3(x) # Now contains CBAM
        x = self.base_model.layer4(x) # Now contains CBAM

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # BNNeck Logic
        feat = self.bottleneck(x) 
        
        # During training, we might want the raw features for Triplet Loss
        # and the logits for Cross Entropy (if you use ID loss).
        # For pure Siamese inference, we want the normalized embedding.
        
        embedding = self.classifier(feat)
        
        # Always normalize embeddings for Triplet Loss
        return F.normalize(embedding, p=2, dim=1)

    def _weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


ReIdModel = LacrosseReIDResNet