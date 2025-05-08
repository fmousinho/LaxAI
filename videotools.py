import os
import logging
import numpy as np
import cv2
import ipywidgets as widgets
from typing import Generator, NamedTuple, Callable, Optional # Import Generator and NamedTuple

logger = logging.getLogger(__name__)

# Define a structure for bounding boxes within this module
class BoundingBox(NamedTuple):
    """Represents a bounding box with x1, y1, width, and height."""
    x1: float
    y1: float
    w: float
    h: float

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> 'BoundingBox':
        """
        Creates a BoundingBox instance from top-left (x1, y1) and bottom-right (x2, y2) coordinates.

        Args:
            x1: Top-left x-coordinate.
            y1: Top-left y-coordinate.
            x2: Bottom-right x-coordinate.
            y2: Bottom-right y-coordinate.
        """
        return cls(x1=x1, y1=y1, w=x2 - x1, h=y2 - y1)
    
    def to_xyxy(self) -> tuple:
        """
        Converts the bounding box to top-left (x1, y1) and bottom-right (x2, y2) coordinates.

        Returns:
            A tuple containing (x1, y1, x2, y2).
        """
        x2 = self.x1 + self.w
        y2 = self.y1 + self.h
        return (self.x1, self.y1, x2, y2)


class VideoToools:
    def __init__(self, input_video_path: str, output_video_path: str):
        """
        Initializes VideoTools by loading the input video and setting up the output video writer.

        Args:
            input_video_path: Path to the input video file.
            output_video_path: Desired path for the output video file.

        Raises:
            FileNotFoundError: If the input video file cannot be opened.
            IOError: If the output video writer cannot be created.
        """
        self.cap = None
        self.out = None
        self.in_fps = None
        self.in_width = None
        self.in_height = None
        self.in_n_frames = None
        self.actual_output_path = None

        # --- Load Input Video ---
        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Error: Could not open input video file: {input_video_path}")

        self.in_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.in_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.in_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.in_n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Input video loaded: {input_video_path} ({self.in_width}x{self.in_height} @ {self.in_fps:.2f} FPS, {self.in_n_frames} frames)")

        # --- Create Output Video File (ensuring unique path) ---
        base_name, extension = os.path.splitext(output_video_path)
        current_output_path = output_video_path
        counter = 1
        while os.path.exists(current_output_path):
            current_output_path = f"{base_name}_{counter}{extension}"
            counter += 1
        self.actual_output_path = current_output_path

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.actual_output_path, fourcc, self.in_fps, (self.in_width, self.in_height))
        if not self.out.isOpened():
            raise IOError(f"Error: Could not open output video writer for path: {self.actual_output_path}")
        logger.info(f"Output video writer created for: {self.actual_output_path}")

    def get_next_frame(self) -> Generator[np.ndarray, None, None]:
        """
        A generator that yields frames sequentially from the input video.

        Returns:
            A generator object yielding each frame as a numpy array. Output is in RGB format.

        Raises:
            IOError: If there's an error reading a frame.
        """
        if not self.cap or not self.cap.isOpened():
            logger.error("Video capture is not initialized or opened.")
            return # Or raise an exception

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Ensure we start from the beginning
        while True:
            ret, frame = self.cap.read()     # Read frame in BGR format
            if not ret:
                logger.info("Reached end of video or error reading frame.")
                break # Exit the loop if no frame is returned
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame


    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draws bounding boxes and labels on the frame for detected objects.

        Args:
            frame: The input video frame as a numpy array.
            detections: A list of detected objects with their bounding boxes and scores.

                        Expected format for each item in detections:
                        (BoundingBox(x1, y1, w, h), confidence, class_id)

        Returns:
            The modified frame with drawn detections.
        """
        for bbox, confidence, class_id in detections:
            # Access attributes directly from the BoundingBox object and convert to int for drawing
            x1, y1, w, h = map(int, [bbox.x1, bbox.y1, bbox.w, bbox.h])
            # Calculate bottom-right coordinates for cv2.rectangle
            x2, y2 = x1 + w, y1 + h
            label = f"Class {class_id} ({confidence:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def draw_tracks(self, frame: np.ndarray, tracks: list,
                    team_id_getter: Optional[Callable[[BoundingBox, np.ndarray], Optional[int]]] = None) -> np.ndarray:
        """
        Draws bounding boxes and track IDs on the frame for tracked objects.
        Colors tracks based on team ID if team_id_getter is provided.

        Args:
            frame: The input video frame as a numpy array (expected in RGB).
            tracks: A list of Track objects from deep_sort_realtime.
            team_id_getter: An optional function that takes a BoundingBox and the current frame,
                            and returns a team ID.

        Returns:
            The modified frame with drawn track information.
        """

        for track in tracks:
            if not track.is_confirmed():
                continue # Skip tracks that are not yet confirmed

            track_id = track.track_id
            ltrb = track.to_ltrb() # Left, Top, Right, Bottom format

            x1, y1, x2, y2 = map(int, ltrb)

            team_id = None
            # Try to get team_id using the original detection bounding box
            if team_id_getter and hasattr(track, 'original_ltwh') and track.original_ltwh is not None:
                original_ltwh = track.original_ltwh # [x, y, w, h]
                # Create BoundingBox from original_ltwh (which is x1,y1,w,h)
                original_detection_bbox = BoundingBox(x1=float(original_ltwh[0]),
                                                      y1=float(original_ltwh[1]),
                                                      w=float(original_ltwh[2]),
                                                      h=float(original_ltwh[3]))
                team_id = team_id_getter(original_detection_bbox, frame) # Pass current frame

            # Determine color based on team_id
            if team_id == 0:
                track_color = (0, 0, 255)   # Red for team 0 (RGB)
            elif team_id == 1:
                track_color = (0, 255, 255) # Yellow for team 1 (RGB) - Changed from blue for better contrast if needed
            else: # Default color if no team or team_id_getter not provided
                track_color = (0, 255, 0)   # Green for unknown/unassigned (RGB)

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 2)
            label = f"ID: {track_id}"
            if team_id is not None:
                label += f" T: {team_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2)
        return frame