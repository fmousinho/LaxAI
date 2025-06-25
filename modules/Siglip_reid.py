import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoProcessor, SiglipVisionModel
import supervision as sv  # type: ignore
from PIL import Image
from collections import defaultdict

from .player import Player

logger = logging.getLogger(__name__)


class ReIDData:
    """Helper class to store data for re-identification."""
    def __init__(self):
        self.crops: List[np.ndarray] = []
        self.embeddings: Optional[np.ndarray] = None

_MIN_HEIGHT_FOR_EMBEDDINGS = 40
_MIN_WIDTH_FOR_EMBEDDINGS = 15
_REID_SIMILARITY_THRESHOLD = 0.85
_EMBEDDINGS_MODEL_PATH = "google/siglip2-base-patch16-224"
_EMBEDDINGS_LEARNING_FREQUENCY = 100  # How often to log embeddings generation progress
_EMBEDDINGS_LEARNING_RATE = 0.1  # Learning rate for embeddings updates

class SiglipReID:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(_EMBEDDINGS_MODEL_PATH)
    model = SiglipVisionModel.from_pretrained(_EMBEDDINGS_MODEL_PATH).to(device)

    def __init__(self):
        """
        Initializes the SiglipReID instance.
        """
        # The original type hint had invalid syntax.
        # Using a defaultdict with a helper class to fix the logic in update_ids.
        self._registry: Dict[int, ReIDData] = defaultdict(ReIDData)

    def update_ids (self, detections: sv.Detections, frame: np.ndarray, frame_id: int):
        """
        Updates and generates embeddings for tracked objects.

        On every frame, this method generates initial embeddings for any new tracker
        ID that appears.

        Periodically (controlled by _EMBEDDINGS_LEARNING_FREQUENCY), it also
        updates the embeddings of all existing, currently visible tracks using an
        exponential moving average. This helps create a more robust representation
        of each track over time.

        Args:
            detections (sv.Detections): The detections from the current frame.
            frame (np.ndarray): The raw image of the current frame, needed for
                cropping images to generate embeddings.
            frame_id (int): The current frame number.
        """
        if detections.tracker_id is None or len(detections.tracker_id) == 0:
            return 
        
        tracker_ids_in_frame = set(detections.tracker_id)

        # --- Periodic Update for Existing Tracks ---
        if frame_id % _EMBEDDINGS_LEARNING_FREQUENCY == 0:
            # Identify existing tracker IDs that are in the frame and already have an embedding
            tids_to_update = {
                tid for tid in tracker_ids_in_frame 
                if self._registry[tid].embeddings is not None
            }

            if tids_to_update:
                logger.debug(f"Frame {frame_id}: Updating embeddings for {len(tids_to_update)} existing tracks.")
                update_mask = np.isin(detections.tracker_id, list(tids_to_update))
                detections_to_update = detections[update_mask]

                update_crops_np = [sv.crop_image(frame, xyxy=xyxy) for xyxy in detections_to_update.xyxy]
                update_crops_pil = [sv.cv2_to_pillow(crop) for crop in update_crops_np]
                latest_embeddings = self._generate_embeddings_for_batch(detections_to_update, update_crops_pil)

                for i, tid_to_update_np in enumerate(detections_to_update.tracker_id):
                    tid = int(tid_to_update_np)
                    new_embedding = latest_embeddings[i]
                    
                    if new_embedding is not None:
                        old_embedding = self._registry[tid].embeddings
                        # Exponential Moving Average to update the embedding
                        updated_embedding = (1 - _EMBEDDINGS_LEARNING_RATE) * old_embedding + _EMBEDDINGS_LEARNING_RATE * new_embedding
                        self._registry[tid].embeddings = updated_embedding
                        self._registry[tid].crops.append(update_crops_np[i]) # Append the latest representative crop

        # --- Initial Embedding Generation for New Tracks ---
        
        new_tracker_ids = {
            tid for tid in tracker_ids_in_frame
            if self._registry[tid].embeddings is None
        }
        
        if not new_tracker_ids:
            return 

        new_detections_mask = np.isin(detections.tracker_id, list(new_tracker_ids))
        new_detections: sv.Detections = detections[new_detections_mask]

        if len(new_detections.xyxy) == 0:
            return 

        crops_np = [sv.crop_image(frame, xyxy=xyxy) for xyxy in new_detections.xyxy]
        crops_pil = [sv.cv2_to_pillow(crop) for crop in crops_np]
        new_embeddings = self._generate_embeddings_for_batch(new_detections, crops_pil)

        for i in range(len(new_detections)):
            tid = int(new_detections.tracker_id[i])
            if new_embeddings[i] is not None:
                self._registry[tid].crops = [crops_np[i]]
                self._registry[tid].embeddings = new_embeddings[i]

        return 


    def _generate_embeddings_for_batch(self, detections: sv.Detections, images: List[Image.Image]) -> List[Optional[np.ndarray]]:
        """
        Generates image embeddings for a batch of detections.

        This method processes a list of images in a single batch, which is much
        more efficient than processing them one by one. For each image, it
        checks if it meets the minimum size requirements. If it does, an
        embedding is generated. If not, the corresponding entry in the returned
        list will be None. The returned list has the same length as the input
        `images` list.

        Args:
            detections (sv.Detections): The detections object, used for context
                                        and length validation.
            images (List[Image.Image]): A list of PIL image crops corresponding
                                        to the detections.

        Returns:
            List[Optional[np.ndarray]]: A list where each element is either a
            NumPy array for the embedding or None if the image was too small.
        """
        if not images:
            return []

        if detections.tracker_id is None or len(detections.tracker_id) != len(images):
            logger.error(f"Batch embedding generation failed: Mismatch between number of detections ({len(detections.tracker_id or [])}) and image crops ({len(images)}).")
            return [None] * len(images)

        # Prepare the list of results, initialized to None
        results: List[Optional[np.ndarray]] = [None] * len(images)

        # Identify valid images and their original indices
        valid_images_with_indices = [
            (i, image) for i, image in enumerate(images)
            if image.height >= _MIN_HEIGHT_FOR_EMBEDDINGS and image.width >= _MIN_WIDTH_FOR_EMBEDDINGS
        ]

        if len(valid_images_with_indices) == 0:
            return results # Return list of Nones

        original_indices, valid_images = zip(*valid_images_with_indices)

        # Generate embeddings for the valid images in a batch
        try:
            inputs = self.processor(images=list(valid_images), return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

        except Exception as e:
            logger.error(f"Failed to generate embeddings in batch: {e}", exc_info=True)
            return results # Return list of Nones on failure

        # Populate the results list at the correct indices
        for i, embedding in enumerate(batch_embeddings):
            original_idx = original_indices[i]
            results[original_idx] = embedding

        return results
    
    def get_embeddings_and_crops_by_tid(self, tid: int) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Retrieves embeddings and representative crops for a given tracker ID.
        Args:
            tid (int): The tracker ID for which to retrieve embeddings and crops. 
        Returns:
            Optional[Tuple[np.ndarray, List[np.ndarray]]]: A tuple containing the embeddings and a list of representative crops.
            Returns None if the tracker ID does not exist or has no embeddings.
        """
        if tid not in self._registry or self._registry[tid].embeddings is None:
            logger.warning(f"Tracker ID {tid} not found or has no embeddings.")
            return None
        
        reid_data = self._registry[tid]
        return reid_data.embeddings, reid_data.crops
