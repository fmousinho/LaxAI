import logging
from typing import Tuple, Optional, Dict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.all_config import model_config, training_config

logger = logging.getLogger(__name__)


class DINOv3Backbone(nn.Module):
    """A thin wrapper around a DINOv3 backbone from timm that exposes
    a feature extractor and (optionally) a small head for producing
    embeddings of size `embedding_dim`.

    This wrapper assumes `timm` is available and a DINOv3 variant name
    is provided via kwargs or `model_config.dino_variant`.
    """

    def __init__(self, embedding_dim: int, dino_variant: Optional[str] = None, dropout: float = 0.0):
        super().__init__()

        try:
            import timm
        except Exception as e:
            raise ImportError(
                "timm is required for the DINOv3 backbone. Install with: pip install timm"
            ) from e

        # Default to the HF repo id for the ConvNeXt-based DINOv3 you provided
        self.dino_variant = dino_variant or getattr(
            model_config, 'dino_variant', 'facebook/dinov3-convnext-small-pretrain-lvd1689m'
        )

        # Create DINOv3 feature extractor. We create the model with no classifier
        # and rely on its feature dimension to attach our embedding head.
        # Many timm DINOv3 entries expose `num_features` and support `pretrained=True`.
        try:
            # If the user provided a HF repo id (owner/name) try to download snapshot first
            if isinstance(self.dino_variant, str) and '/' in self.dino_variant:
                try:
                    from huggingface_hub import snapshot_download
                    # use env token if available (huggingface_hub will read HUGGINGFACE_HUB_TOKEN)
                    snapshot = snapshot_download(repo_id=self.dino_variant, cache_dir='models/dinov3')
                    # try to locate a model file in snapshot
                    import os
                    model_path = None
                    for candidate in ('model.safetensors', 'pytorch_model.bin', 'pytorch_model.pt'):
                        p = os.path.join(snapshot, candidate)
                        if os.path.exists(p):
                            model_path = p
                            break
                    # First try to load via Hugging Face transformers if available
                    tf_success = False
                    try:
                        from transformers import AutoModel, AutoConfig, AutoImageProcessor
                        try:
                            model = AutoModel.from_pretrained(snapshot, trust_remote_code=True)
                            self.backbone = model
                            self._is_transformers = True
                            # load processor if available
                            try:
                                self._image_processor = AutoImageProcessor.from_pretrained(snapshot)
                            except Exception:
                                self._image_processor = None
                            tf_success = True
                            logger.info(f'Loaded Hugging Face model from snapshot {snapshot}')
                        except Exception:
                            logger.warning('transformers present but failed to load HF model; will fallback')
                    except Exception:
                        # transformers not installed or failed import
                        self._is_transformers = False

                    if not tf_success:
                        # prefer to create a reasonable timm ConvNeXt model and try to load local weights
                        # infer timm model name from variant or config (convnext small is a good guess)
                        timm_name = 'convnext_small'
                        try:
                            self.backbone = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool='')
                            if model_path is not None:
                                try:
                                    # try safetensors first
                                    try:
                                        from safetensors.torch import load_file as safetensors_load
                                        state = safetensors_load(model_path)
                                        self.backbone.load_state_dict(state, strict=False)
                                        logger.info(f'Loaded weights from {model_path} via safetensors')
                                    except Exception:
                                        # fallback to torch load
                                        import torch as _torch
                                        state = _torch.load(model_path, map_location='cpu')
                                        if isinstance(state, dict) and 'state_dict' in state:
                                            state = state['state_dict']
                                        new_state = {}
                                        for k,v in state.items():
                                            nk = k
                                            if k.startswith('module.'):
                                                nk = k[len('module.'):]
                                            new_state[nk] = v
                                        self.backbone.load_state_dict(new_state, strict=False)
                                        logger.info(f'Loaded weights from {model_path} via torch')
                                except Exception:
                                    logger.exception('Failed to load local weights into timm model; continuing with model defaults')
                        except Exception:
                            # if timm cannot create convnext_small, fall back to attempting to create by variant name
                            try:
                                self.backbone = timm.create_model(self.dino_variant.split('/')[-1], pretrained=False, num_classes=0, global_pool='')
                            except Exception:
                                raise
                except Exception:
                    logger.warning('Could not snapshot download HF repo; falling back to timm.create_model with pretrained=True')
                    self.backbone = timm.create_model(self.dino_variant, pretrained=True, num_classes=0, global_pool='')
            else:
                # use num_classes=0 or 0/'' depending on timm; use `features_only` when available
                # prefer create_model with pretrained weights
                self.backbone = timm.create_model(self.dino_variant, pretrained=True, num_classes=0, global_pool='')
        except TypeError:
            # older timm variants may not accept num_classes=0; try without
            self.backbone = timm.create_model(self.dino_variant, pretrained=True)

        # Infer feature dimension. timm models commonly expose `num_features` or `feature_info`
        if hasattr(self.backbone, 'num_features') and getattr(self.backbone, 'num_features'):
            nf = getattr(self.backbone, 'num_features')
            if isinstance(nf, int):
                self._feat_dim = nf
            elif isinstance(nf, torch.Tensor):
                try:
                    self._feat_dim = int(nf.item())
                except Exception:
                    self._feat_dim = None
            else:
                # unknown type (module or other), skip and fallback
                self._feat_dim = None
        else:
            # fallback attempt: try to infer from classifier if present
            feat_dim = getattr(self.backbone, 'embed_dim', None) or getattr(self.backbone, 'head_dim', None)
            if feat_dim is not None:
                try:
                    self._feat_dim = int(feat_dim)
                except Exception:
                    self._feat_dim = None
            else:
                # final attempt: inspect a forward with dummy input later (lazy)
                self._feat_dim = None

        # Embedding head will be created lazily on first forward if feat_dim is unknown now
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self._head = None

        # container for attention maps captured via hooks (best-effort)
        self._attn_maps = {}
        self._registered_hooks = False

        # determine model expected input size (height, width)
        expected_h = None
        expected_w = None
        try:
            if hasattr(self, '_image_processor') and self._image_processor is not None:
                size = getattr(self._image_processor, 'size', None)
                if isinstance(size, dict) and 'height' in size and 'width' in size:
                    expected_h = int(size['height'])
                    expected_w = int(size['width'])
        except Exception:
            expected_h = None
            expected_w = None

        if expected_h is None or expected_w is None:
            # fallback default commonly used by DINOv3 families
            expected_h, expected_w = 224, 224

        self._expected_input_size = (expected_h, expected_w)

        # If the project config provides a different input size, add a resize adapter
        cfg_h = getattr(model_config, 'input_height', None)
        cfg_w = getattr(model_config, 'input_width', None)
        if cfg_h is not None and cfg_w is not None and (cfg_h, cfg_w) != self._expected_input_size:
            # create a simple resize module (channels-first tensors expected)
            self._input_adapter = nn.Upsample(size=self._expected_input_size, mode='bilinear', align_corners=False)
            logger.info(f'Input adapter created: resizing from config size ({cfg_h},{cfg_w}) to model expected {self._expected_input_size}')
        else:
            self._input_adapter = None

        logger.info(f"Initialized DINOv3Backbone variant={self.dino_variant}, pretrained=True")

    def _build_head(self, feat_dim: int):
        head = nn.Sequential(
            nn.Dropout(self.dropout) if self.dropout and self.dropout > 0.0 else nn.Identity(),
            nn.Linear(feat_dim, self.embedding_dim)
        )
        return head

    def _ensure_head(self, x: Optional[torch.Tensor] = None):
        if self._head is not None and self._feat_dim is not None:
            return
        if self._feat_dim is None:
            if x is None:
                raise RuntimeError("Cannot infer backbone feature dimension without example input")
            # run a dry-forward to infer feature dim
            with torch.no_grad():
                feats = self._forward_features(x)
            feat_dim = int(feats.shape[-1])
            self._feat_dim = feat_dim

        # create head
        # ensure we pass an int
        self._head = self._build_head(int(self._feat_dim))

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # apply input adapter if present (expects channels-first tensor)
        if hasattr(self, '_input_adapter') and self._input_adapter is not None:
            # model expects (C,H,W) per sample; x is (B,C,H,W)
            _, _, h, w = x.shape
            exp_h, exp_w = self._expected_input_size
            if (h, w) != (exp_h, exp_w):
                # resize using bilinear on CPU/GPU as appropriate
                x = self._input_adapter(x)

        # many timm models provide a `forward_features` method
        func = getattr(self.backbone, 'forward_features', None)
        if callable(func):
            out = cast(torch.Tensor, func(x))
            # if spatial feature map, global-pool to (B, C)
            if out.ndim == 4:
                out = out.mean((-2, -1))
            elif out.ndim == 3:
                # sequence-like output: mean over sequence dim
                out = out.mean(1)
            return out
        # otherwise call the model and try to strip off a classifier if present
        out = self.backbone(x)
        if out.ndim == 4:
            # global pool spatial dims
            out = out.mean((-2, -1))
        elif out.ndim == 3:
            out = out.mean(1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._forward_features(x)
        # lazy head creation
        if self._head is None:
            self._ensure_head(x)
            # move head to same device as backbone
            device = next(self.backbone.parameters()).device
            if self._head is not None:
                self._head = self._head.to(device)
            else:
                raise RuntimeError("Failed to initialize embedding head")

        emb = self._head(feats)
        return emb

    def clear_attention_cache(self):
        self._attn_maps = {}

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Return any captured attention maps; this is best-effort and may be empty depending
        on whether the underlying timm model exposes attention internals.
        """
        return dict(self._attn_maps)



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
        # dino variant name can be provided via kwargs or model_config
        dino_variant = kwargs.get('dino_variant', getattr(model_config, 'dino_variant', None))

        # instantiate DINOv3 backbone (requires timm)
        self.backbone = DINOv3Backbone(self.embedding_dim, dino_variant=dino_variant, dropout=self.dropout_rate)

        logger.info(f"SiameseNet initialized with DINOv3 backbone variant={self.backbone.dino_variant}")
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
