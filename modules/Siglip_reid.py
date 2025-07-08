import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from more_itertools import chunked
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
import supervision as sv  # type: ignore
from PIL import Image
from umap import UMAP
import cv2
from collections import defaultdict

logger = logging.getLogger(__name__)


_EMBEDDINGS_MODEL_PATH = "google/siglip2-base-patch16-224"
_BATCH_SIZE = 32
_TOP_CROP_FACTOR = 0.1
_BOTTOM_CROP_FACTOR = 0.4
_LEFT_CROP_FACTOR = .1
_RIGHT_CROP_FACTOR = .1


class SiglipReID:

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(_EMBEDDINGS_MODEL_PATH)
    model = SiglipVisionModel.from_pretrained(_EMBEDDINGS_MODEL_PATH).to(device)

    def __init__(self):
        """
        Initializes the SiglipReID instance.
        """
        self.umap_model = UMAP(n_components=3, random_state=42)

    def _get_roi_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts a region of interest (ROI) crop from the given image crop.

        Args:
            crop (np.ndarray): The input image crop.

        Returns:
            np.ndarray: The ROI crop.
        """
        crop = crop.copy()
        height, width = crop.shape[:2]
        top = int(height * _TOP_CROP_FACTOR)
        bottom = int(height * _BOTTOM_CROP_FACTOR)
        left = int(width * _LEFT_CROP_FACTOR)
        right = int(width * _RIGHT_CROP_FACTOR)

        if top >= bottom or left >= right:
            return crop

        roi_crop = crop[top:bottom, left:right]
        return roi_crop

    def get_emb_from_crops(self, crops: List[np.ndarray], format: str = "BGR") -> np.ndarray:
        """
        Generates SigLip embeddings for a list of crops.

        Args:
            crops (List[np.ndarray]): A list of np.array image crops.

        Returns:
            np.ndarray: Embeddings for each crop, in an array with dimensions: 
            len(crops) x len(Siglip embedding size [768]).
        """

        if not crops: return np.empty((0, 768))
        crops = crops.copy()  # Avoid modifying the original list
       
        if format in ["BGR"]:
            crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
        else:
            return np.empty((0, 768))  # Unsupported format
    

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
        results = self.umap_model.fit_transform(results)

        return results
    
    def get_avg_color_from_crops(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Computes the average color for each crop in the list.

        Args:
            crops (List[np.ndarray]): A list of np.array image crops.

        Returns:
            np.ndarray: An array of average colors for each crop.
        """
        if not crops: return np.empty((0, 3))

        avg_colors = []
        crops = crops.copy()
        for crop in crops:
            crop = self._get_roi_crop(crop)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            if crop.size == 0:
                avg_colors.append(np.empty((0, 3), dtype=np.float32))
                logger.warning("Empty crop encountered, skipping.")
                continue
            avg_color = cv2.mean(crop)[:3]
            avg_colors.append(avg_color)

        return np.array(avg_colors)
