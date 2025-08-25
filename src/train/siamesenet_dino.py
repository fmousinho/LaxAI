import logging
import os
from typing import Tuple, Optional, Dict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModel

from config.all_config import model_config, training_config

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
        
        # require HF token for gated model access
        token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
        if not token:
            raise RuntimeError('HUGGINGFACE_HUB_TOKEN environment variable is not set')
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract_features(x)
        
        # lazy head creation on first forward
        if self._head is None:
            feat_dim = int(feats.shape[-1])
            self._head = self._build_head(feat_dim)
            # move head to same device as backbone
            device = next(self.backbone.parameters()).device
            self._head = self._head.to(device)

        emb = self._head(feats)
        return emb


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

        Note: capturing transformer attention internals is model-dependent. This
        implementation is best-effort; if the underlying timm model exposes
        attention maps/hooks they will be returned, otherwise an empty dict is
        returned and a warning is logged.
        """
        # If an input was provided and the backbone hasn't registered hooks, try a forward pass
        if x is not None:
            # forward to populate any lazy structures
            with torch.no_grad():
                _ = self.backbone(x)

        attn = self.backbone.get_attention_maps() if hasattr(self.backbone, 'get_attention_maps') else {}
        if not attn:
            logger.warning("No attention maps available from DINOv3 backbone (model may not expose internals).")
        return attn

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
