import logging
import numpy as np
import torch
from . import constants as const
from .store_driver import Store
from .videotools import VideoToools # Import VideoToools
from typing import Optional
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter
import cv2 # Import cv2 for color conversion

logger = logging.getLogger(__name__)

class DetectionModel:
    """Handles the loading and inference of the object detection model."""

    def __init__(self, 
                 store: Store,
                 model_name: Optional[str] = None,
                 drive_path: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 player_class_id: int = const.DEFAULT_PLAYER_CLASS_ID,
                 writer: Optional[SummaryWriter] = None, # Add writer
                 tools: Optional[VideoToools] = None):  # Add tools
        """
        Initializes the DetectionModel.

        Args:
            model_name: The name of the model file on Google Drive.
            drive_path: The path within Google Drive where the model file resides. Defaults to const.GOOGLE_DRIVE_PATH.
            device: The torch.device (cpu or cuda) to load the model onto.
            store: An initialized Store object for Google Drive access. This is required for model loading from Drive.
            player_class_id: The class ID that represents players in the detection model.
            writer: An optional TensorBoard SummaryWriter instance for logging.
            tools: An optional VideoToools instance for drawing detections.
        """
        self.model = None  # Model will be loaded later
        self.model_name = model_name if model_name is not None else const.DEFAULT_DETECTION_MODEL
        self.drive_path = drive_path if drive_path is not None else const.DEFAULT_MODEL_DRIVE_FOLDER
        self.device = device if device is not None else torch.device("cpu")
        self.player_class_id = player_class_id
        self.writer = writer
        self.tools = tools # Store the VideoToools instance
        self.debug_detection_frame_sampling_rate = const.DEBUG_DETECTION_FRAME_SAMPLING_RATE
        
        if store:
            # Attempt to load the model immediately upon initialization
            if not self._load_from_drive(store):
                 raise RuntimeError(f"Failed to load detection model '{self.model_name}' from Drive path '{self.drive_path}'.")
        else:
            raise ValueError("Store object is required for DetectionModel initialization to load the model from Drive.")
        logger.info(f"DetectionModel initialized and model '{self.model_name}' loaded successfully.")

    def _load_from_drive(self, store: Store) -> bool:
        """
        Downloads the model file from Google Drive and loads it onto the specified device.

        Args:
            store: An initialized Store object for Google Drive access.

        Returns:
            True if the model was loaded successfully, False otherwise.
        """
        logger.info(f"Attempting to download detection model '{self.model_name}' from Google Drive path '{self.drive_path}'...")
        model_buffer = store.download_file_by_name(self.model_name, self.drive_path)

        if model_buffer:
            logger.info(f"Detection model file '{self.model_name}' downloaded successfully. Loading into PyTorch...")
            try:
                # Ensure custom classes needed for unpickling are added
                # This is for unpickling if the model's saved structure refers to classes by this name.
                # Removed unnecessary add_safe_globals call
                self.model = torch.load(model_buffer, map_location=self.device, weights_only=False)
                logger.info(f"Detection model loaded successfully to {self.device}")
                return True
            except Exception as e:
                logger.error(f"Error loading detection model from downloaded buffer: {e}", exc_info=True)
                self.model = None # Ensure model is None if loading fails
                return False
        else:
            logger.error(f"Failed to download detection model '{self.model_name}' from Google Drive. Cannot proceed.")
            self.model = None
            return False

    def generate_detections(self, frame_in_rgb: np.ndarray, frame_idx: int, threshold: float = const.DEFAULT_DETECTION_THRESHOLD) -> list:
        """
        Runs inference on a single frame using the loaded detection model.
        Also handles debug logging of detections to TensorBoard.

        Args:
            frame_in_rgb: The input video frame as a numpy array (expected in RGB).
            frame_idx: The current frame index, used for conditional logging.
            threshold: Confidence threshold for detections.

        Returns:
            A list of detected objects with their bounding boxes and scores.
            Expected format for DeepSort: [(bbox list, confidence, class)]
            bbox list is in xywh format [x, y, w, h].
        """
        if self.model is None:
            logger.error("Detection model is not loaded. Cannot perform detection.")
            return []
        
        # Assuming self.model.predict returns an object with .xyxy, .confidence, .class_id
        # and .xyxy is in [x1, y1, x2, y2] format
        model_output_detections = self.model.predict(frame_in_rgb, device=self.device, threshold=threshold)

        formatted_detections = []
        # Iterate through detections and format for DeepSort
        # Ensure detections object has expected attributes and is iterable
        if not (hasattr(model_output_detections, 'xyxy') and
                  hasattr(model_output_detections, 'confidence') and
                  hasattr(model_output_detections, 'class_id')):
            logger.warning("DetectionModel: Model output does not have expected attributes (xyxy, confidence, class_id).")
            return formatted_detections

        # Ensure the attributes are not None and are iterable (or have a length)
        if model_output_detections.xyxy is None or \
           model_output_detections.confidence is None or \
           model_output_detections.class_id is None:
            logger.warning("DetectionModel: One or more model output attributes (xyxy, confidence, class_id) are None.")
            return formatted_detections

        num_model_detections = len(model_output_detections.xyxy)
        if not (len(model_output_detections.confidence) == num_model_detections and \
                  len(model_output_detections.class_id) == num_model_detections):
            logger.warning("DetectionModel: Mismatch in lengths of model output attributes (xyxy, confidence, class_id).")
            return formatted_detections

        for i in range(num_model_detections):
            if int(model_output_detections.class_id[i]) != self.player_class_id:
                continue

            # Convert xyxy [x1, y1, x2, y2] to xywh [x, y, w, h] for DeepSort
            x1, y1, x2, y2 = map(int, model_output_detections.xyxy[i]) # Ensure integer coordinates
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0:
                logger.debug(f"DetectionModel: Skipping detection with invalid dimensions: w={w}, h={h} for xyxy={model_output_detections.xyxy[i]}")
                continue

            formatted_detections.append(
                ([x1, y1, w, h],
                 float(model_output_detections.confidence[i]),
                 int(model_output_detections.class_id[i]))
            )
        
        # --- Debug Logging: Log frame with detections to TensorBoard ---
        if logger.isEnabledFor(logging.DEBUG) and self.writer and self.tools and \
           frame_idx % self.debug_detection_frame_sampling_rate == 0:
            try:
                # self.tools.draw_detections expects BGR, so convert frame_in_rgb temporarily
                frame_bgr_for_drawing = cv2.cvtColor(frame_in_rgb.copy(), cv2.COLOR_RGB2BGR)
                # formatted_detections is already in the format expected by draw_detections
                frame_with_detections_bgr = self.tools.draw_detections(frame_bgr_for_drawing, formatted_detections)
                
                frame_with_detections_rgb = cv2.cvtColor(frame_with_detections_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor_chw_float_rgb = torch.from_numpy(frame_with_detections_rgb).permute(2, 0, 1).float() / 255.0
                
                self.writer.add_image(f"Detections/Frame_{frame_idx}", frame_tensor_chw_float_rgb, global_step=frame_idx)
                logger.debug(f"DetectionModel: Logged frame {frame_idx} with {len(formatted_detections)} detections to TensorBoard.")
            except Exception as e:
                logger.error(f"DetectionModel: Error logging frame {frame_idx} with detections: {e}", exc_info=True)

        logger.debug(f"DetectionModel: Generated {len(formatted_detections)} player detections for frame {frame_idx}.")
        return formatted_detections
