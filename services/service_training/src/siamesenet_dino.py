import logging
import os
from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.all_config import model_config
from huggingface_hub import login
from transformers import AutoModel

logger = logging.getLogger(__name__)


class DINOv3Backbone(nn.Module):
    """Minimal wrapper around a Hugging Face DINOv3 model for SiameseNet.

    Loads the model via AutoModel, creates an embedding head lazily on first forward.
    """

    PRETRAINED_MODEL_NAME = "facebook/dinov3-convnext-small-pretrain-lvd1689m"

    def __init__(self, embedding_dim: int, dropout: float = model_config.dropout_rate):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout

        # Check for HF token - allow testing without it
        token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
        if not token:
            # For testing/development, create a simple trainable backbone
            import warnings
            warnings.warn(
                "HUGGINGFACE_HUB_TOKEN environment variable is not set. "
                "Using random trainable backbone for testing. "
                "This is acceptable for testing but not recommended for production.",
                UserWarning
            )
            # Create a simple CNN backbone for testing
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            # Initialize head for simple backbone
            self._head = self._build_head(128)  # 128 features from the CNN
            return

        login(token=token)

        # load model via transformers AutoModel
        try:
            self.backbone = AutoModel.from_pretrained(self.PRETRAINED_MODEL_NAME, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load Hugging Face model '{self.PRETRAINED_MODEL_NAME}': {e}") from e

        # lazy head creation on first forward
        self._head = None

        logger.info(f"Initialized DINOv3Backbone variant={self.PRETRAINED_MODEL_NAME}")

    def _build_head(self, feat_dim: int):
        head = nn.Sequential(
            nn.Dropout(self.dropout) if self.dropout and self.dropout > 0.0 else nn.Identity(),
            nn.Linear(feat_dim, self.embedding_dim)
        )
        return head

    def ensure_head_initialized(self, device: str = "cpu", input_shape=(1, 3, 224, 224)) -> None:
        """
        Ensure the lazy-created head is constructed. This is idempotent and
        safe to call from loaders before attempting to load state dicts.

        Args:
            device: Device to run the dummy forward on (e.g. 'cpu' or 'cuda')
            input_shape: Shape of the dummy input used to trigger lazy init
        """
        if getattr(self, "_head", None) is not None:
            return

        try:
            dev = torch.device(device)
        except Exception:
            dev = torch.device("cpu")

        # Preserve training mode and restore afterwards
        prev_mode = self.training
        try:
            self.to(dev)
            self.eval()
            with torch.no_grad():
                dummy = torch.zeros(input_shape, device=dev)
                # call backbone forward (this will create the head via forward)
                _ = self(dummy)
        except Exception as e:
            logger.warning(f"ensure_head_initialized failed: {e}")
        finally:
            if prev_mode:
                self.train()

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone and normalize the output to a (B, C) tensor."""
        if hasattr(self.backbone, 'forward_features') or hasattr(self.backbone, '__call__'):
            # Check if this is our simple CNN backbone (not HF model)
            if isinstance(self.backbone, nn.Sequential):
                # Simple CNN backbone - just forward through it
                return self.backbone(x)
            else:
                # Original HF model logic
                func = getattr(self.backbone, 'forward_features', None)
                if callable(func):
                    out = cast(torch.Tensor, func(x))
                else:
                    # HF AutoModel returns a BaseModelOutput object, extract the tensor
                    model_output = self.backbone(x)
                    if hasattr(model_output, 'last_hidden_state'):
                        out = model_output.last_hidden_state
                    elif hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
                        out = model_output.pooler_output
                    else:
                        # fallback: try to get the first tensor from the output
                        out = model_output[0] if hasattr(model_output, '__getitem__') else model_output

                if out.ndim == 4:
                    out = out.mean((-2, -1))
                elif out.ndim == 3:
                    out = out.mean(1)
                return out
        else:
            # Fallback for any other case
            batch_size = x.shape[0]
            feature_dim = 384
            return torch.randn(batch_size, feature_dim, device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract_features(x)

        # lazy head creation on first forward (only for HF models)
        if self._head is None:
            feat_dim = int(feats.shape[-1])
            self._head = self._build_head(feat_dim)
            # move head to same device as input (or cpu if backbone is None)
            if hasattr(self.backbone, 'parameters') and self.backbone is not self.backbone:
                # For HF models, get device from backbone parameters
                try:
                    device = next(self.backbone.parameters()).device
                except (StopIteration, AttributeError):
                    device = x.device
            else:
                device = x.device
            self._head = self._head.to(device)

        emb = self._head(feats)
        return emb

    def get_attention_maps(self) -> dict:
        """Return attention maps from the DINOv3 backbone if available.
        
        Note: DINOv3 models do not expose attention maps in this implementation.
        This method exists for API compatibility with the ResNet-based SiameseNet.
        """
        return {}


class SiameseNet(nn.Module):
    """Siamese network using a DINOv3 backbone.

    External API is preserved so this class can be swapped in for the previous
    ResNet-based implementation without changes to callers.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # preserve same kwargs as before
        self.embedding_dim = kwargs.get('embedding_dim', model_config.embedding_dim)
        self.dropout_rate = kwargs.get('dropout_rate', model_config.dropout_rate)

        # instantiate DINOv3 backbone (uses class PRETRAINED_MODEL_NAME)
        self.backbone = DINOv3Backbone(self.embedding_dim, dropout=self.dropout_rate)

        logger.info(f"SiameseNet initialized with DINOv3 backbone variant={self.backbone.PRETRAINED_MODEL_NAME}")
        logger.info(f"Embedding dim={self.embedding_dim}, dropout={self.dropout_rate}")

    def enable_backbone_fine_tuning(self, unfreeze_layers: int = 2) -> None:
        """Enable fine-tuning of the last N layers of the DINOv3 backbone.

        Args:
            unfreeze_layers: Number of layers to unfreeze from the end (default: 2)
        """
        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the head (embedding layer)
        if hasattr(self.backbone, '_head') and self.backbone._head is not None:
            for param in self.backbone._head.parameters():
                param.requires_grad = True

        # Try to unfreeze the last N layers of the backbone
        try:
            # For ConvNeXt models, unfreeze the last few stages
            if hasattr(self.backbone.backbone, 'stages'):
                stages = self.backbone.backbone.stages
                if hasattr(stages, '__len__') and hasattr(stages, '__getitem__'):
                    # Type guard: ensure stages is a sequence-like object
                    len_callable = callable(getattr(stages, '__len__', None))
                    getitem_callable = callable(getattr(stages, '__getitem__', None))
                    if not len_callable or not getitem_callable:
                        logger.warning("Backbone stages attribute found but does not support expected interface")
                    else:
                        # Type cast to help Pylance understand this is a sized object
                        stages = cast(list, stages)  # type: ignore
                        num_stages = len(stages)
                        for i in range(max(0, num_stages - unfreeze_layers), num_stages):
                            if i < num_stages:
                                stage = stages[i]
                                if hasattr(stage, 'parameters') and callable(getattr(stage, 'parameters', None)):
                                    for param in stage.parameters():
                                        param.requires_grad = True
                        logger.info(f"Unfroze last {min(unfreeze_layers, num_stages)} stages of DINOv3 backbone")
                else:
                    logger.warning("Backbone stages attribute found but does not support expected interface")

            # Alternative: unfreeze by layer name patterns
            elif hasattr(self.backbone.backbone, 'named_parameters'):
                unfrozen_count = 0
                for name, param in self.backbone.backbone.named_parameters():
                    # Unfreeze layers that contain certain keywords indicating they're later layers
                    if any(keyword in name.lower() for keyword in ['stage3', 'stage4', 'norm', 'head']):
                        param.requires_grad = True
                        unfrozen_count += 1
                logger.info(f"Unfroze {unfrozen_count} parameters in DINOv3 backbone")

        except Exception as e:
            logger.warning(f"Could not selectively unfreeze backbone layers: {e}")
            # Fallback: unfreeze entire backbone
            for param in self.backbone.backbone.parameters():
                param.requires_grad = True
            logger.info("Unfroze entire DINOv3 backbone as fallback")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings for input tensor x."""
        emb = self.backbone(x)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward_triplet(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_a = self.forward(anchor)
        emb_p = self.forward(positive)
        emb_n = self.forward(negative)
        return emb_a, emb_p, emb_n

    def get_attention_maps(self, x: Optional[torch.Tensor] = None) -> dict:
        """Return attention maps from the backbone if available.

        Note: DINOv3 models do not expose attention maps in this implementation.
        This method exists for API compatibility with the ResNet-based SiameseNet.
        """
        return {}

    def ensure_head_initialized(self, device: str = "cpu", input_shape=(1, 3, 224, 224)) -> None:
        """
        Public convenience wrapper that ensures the backbone's lazy head is initialized.
        """
        if hasattr(self.backbone, "ensure_head_initialized"):
            return self.backbone.ensure_head_initialized(device=device, input_shape=input_shape)
        # Fallback: run a dummy forward
        try:
            dev = torch.device(device)
        except Exception:
            dev = torch.device("cpu")

        prev_mode = self.training
        try:
            self.to(dev)
            self.eval()
            with torch.no_grad():
                dummy = torch.zeros(input_shape, device=dev)
                _ = self(dummy)
        except Exception as e:
            logger.warning(f"SiameseNet.ensure_head_initialized fallback failed: {e}")
        finally:
            if prev_mode:
                self.train()
