import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._api import WeightsEnum

from shared_libs.config.all_config import model_config

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_planes // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_planes, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        scale = torch.sigmoid(out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.bn(self.conv(x_cat)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Applies Channel Attention followed by Spatial Attention.
    """

    def __init__(self, channels: int, reduction: int = 16, sa_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(sa_kernel)
        # For lightweight inspection
        self._last_channel_attn: Optional[torch.Tensor] = None
        self._last_spatial_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ca(x)
        # Save channel attention map (broadcasted scale) for the first item
        try:
            with torch.no_grad():
                b, c, h, w = x.size()
                # Recompute the scale used inside ChannelAttention for logging only
                # Keep it lightweight: global avg and max pools
                avg_scale = torch.mean(x, dim=(2, 3), keepdim=False)
                max_scale, _ = torch.max(x.view(b, c, -1), dim=2)
                scale = torch.sigmoid(avg_scale + max_scale).view(b, c, 1, 1)
                self._last_channel_attn = scale[0].detach().cpu()
        except Exception:
            self._last_channel_attn = None

        out = self.sa(out)
        try:
            with torch.no_grad():
                avg_out = torch.mean(out, dim=1, keepdim=True)
                max_out, _ = torch.max(out, dim=1, keepdim=True)
                x_cat = torch.cat([avg_out, max_out], dim=1)
                attn = torch.sigmoid(self.sa.bn(self.sa.conv(x_cat)))
                self._last_spatial_attn = attn[0].detach().cpu()
        except Exception:
            self._last_spatial_attn = None

        return out

    def last_attention_maps(self) -> dict:
        maps = {}
        if self._last_channel_attn is not None:
            maps["channel"] = self._last_channel_attn  # (C,1,1)
        if self._last_spatial_attn is not None:
            maps["spatial"] = self._last_spatial_attn  # (1,H,W)
        return maps


class ResNet50BackboneCBAM(nn.Module):
    """ResNet50 backbone with a CBAM block after layer2."""

    def __init__(self, embedding_dim: int, dropout: float, pretrained: bool = False):
        super().__init__()

        # Safe pretrained toggle (avoid downloads in restricted environments)
        try:
            model = resnet50(weights=(resnet50.weights.DEFAULT if pretrained else None))  # type: ignore[attr-defined]
        except Exception:
            model = resnet50(weights=None)

        # Keep feature extractor components
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        feat_dim = model.fc.in_features  # 2048 for resnet50
        cbam_channels = getattr(self.layer2[-1], "conv3", None)
        if cbam_channels is not None:
            cbam_channels = cbam_channels.out_channels
        else:
            # Fallback to known ResNet50 channel width after layer2
            cbam_channels = 512

        self.cbam = CBAM(channels=cbam_channels)

        head = []
        if dropout and dropout > 0.0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(feat_dim, embedding_dim))
        self.head = nn.Sequential(*head)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cbam(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Global pool input has feat_dim channels
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features just before avgpool to allow CBAM over the conv map
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # Apply CBAM after layer2 to modulate mid-level features
        x = self.cbam(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.head(x)
        return emb

    def get_attention_maps(self) -> dict:
        return self.cbam.last_attention_maps()


class SiameseNet(nn.Module):
    """Siamese network using a ResNet50 backbone with CBAM.

    API-compatible with the DINOv3-based SiameseNet.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.embedding_dim = kwargs.get("embedding_dim", getattr(model_config, "embedding_dim", 256))
        self.dropout_rate = kwargs.get("dropout_rate", getattr(model_config, "dropout_rate", 0.1))
        self.num_classes = kwargs.get("num_classes", None)
        self.pretrained = kwargs.get("pretrained", False)

        self.backbone = ResNet50BackboneCBAM(
            embedding_dim=self.embedding_dim, dropout=self.dropout_rate, pretrained=self.pretrained
        )

        # Optional classification head
        self.classification_head: Optional[nn.Module] = None
        if self.num_classes is not None and int(self.num_classes) > 0:
            self.classification_head = nn.Sequential(
                nn.Dropout(self.dropout_rate), nn.Linear(self.embedding_dim, int(self.num_classes))
            )
            logger.info(f"Classification head initialized with {self.num_classes} classes")

        logger.info(
            f"SiameseNet(ResNet50+CBAM) initialized: emb_dim={self.embedding_dim}, dropout={self.dropout_rate}, pretrained={self.pretrained}"
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def enable_backbone_fine_tuning(self, unfreeze_layers: int = 2) -> None:
        """Enable fine-tuning of the last N layers of the ResNet50 backbone."""
        # Freeze all
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze head
        for p in self.backbone.head.parameters():
            p.requires_grad = True

        # Unfreeze last N residual layers (layer4, then layer3, ...)
        layers = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
        to_unfreeze = layers[: max(0, min(unfreeze_layers, len(layers)))]
        unfrozen = 0
        for layer in to_unfreeze:
            for p in layer.parameters():
                p.requires_grad = True
                unfrozen += 1
        logger.info(f"Unfroze parameters in the last {len(to_unfreeze)} resnet stages (params: {unfrozen})")

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        emb = self.backbone(x)
        emb_norm = F.normalize(emb, p=2, dim=1)
        if return_logits and self.classification_head is not None:
            logits = self.classification_head(emb)
            return emb_norm, logits
        return emb_norm

    def forward_triplet(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if return_logits and self.classification_head is not None:
            ea, la = self.forward(anchor, return_logits=True)
            ep, lp = self.forward(positive, return_logits=True)
            en, ln = self.forward(negative, return_logits=True)
            return (ea, la), (ep, lp), (en, ln)
        else:
            ea = self.forward(anchor)
            ep = self.forward(positive)
            en = self.forward(negative)
            return ea, ep, en

    def get_attention_maps(self, x: Optional[torch.Tensor] = None) -> dict:
        # Returns last computed CBAM maps; if x is provided, run a forward to refresh
        if x is not None:
            with torch.no_grad():
                _ = self.backbone(x)
        return self.backbone.get_attention_maps()

    def ensure_head_initialized(self, device: str = "cpu", input_shape=(1, 3, 224, 224)) -> None:
        # No-op for this backbone since head is constructed eagerly
        return None
