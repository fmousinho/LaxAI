import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from more_itertools import chunked
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
import supervision as sv  # type: ignore
from PIL import Image
from collections import defaultdict

logger = logging.getLogger(__name__)


_EMBEDDINGS_MODEL_PATH = "google/siglip2-base-patch16-224"
_BATCH_SIZE = 32

class SiglipReID:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(_EMBEDDINGS_MODEL_PATH)
    model = SiglipVisionModel.from_pretrained(_EMBEDDINGS_MODEL_PATH).to(device)

    def __init__(self):
        """
        Initializes the SiglipReID instance.
        """

    def get_emb_from_crops(self, crops: List[np.ndarray], format: str = "BGR") -> np.ndarray:
        """
        Generates embeddings for a list of crops.

        Args:
            crops (List[np.ndarray]): A list of np.array image crops.

        Returns:
            np.ndarray: Embeddings for each crop, in an array with dimensions: 
            len(crops) x len(Siglip embedding size [768]).
        """

        if not crops: return np.empty((0, 768))
        if format in ["BGR"]:
            crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        elif format in ["Pillow", "PIL"]:
            crops_pil = crops

        batches = chunked(crops_pil, _BATCH_SIZE)
        data = []

        total_batches = len(crops_pil) // _BATCH_SIZE + 1
        for batch in tqdm(batches, desc="Generating embeddings", total=total_batches):
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize each embedding
            data.append(embeddings)

        results = np.concatenate(data)

        return results
