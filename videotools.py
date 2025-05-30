import os
import logging
import numpy as np
import cv2
import base64 
from typing import Generator, NamedTuple, Callable, Optional # Import Generator and NamedTuple

logger = logging.getLogger(__name__)

# Define a structure for bounding boxes within this module
class BoundingBox(NamedTuple):
    """Represents a bounding box with x1, y1, width, and height."""
    x1: int
    y1: int
    w: int
    h: int

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> 'BoundingBox':
        """
        Creates a BoundingBox instance from top-left (x1, y1) and bottom-right (x2, y2) coordinates.

        Args:
            x1: Top-left x-coordinate.
            y1: Top-left y-coordinate.
            x2: Bottom-right x-coordinate.
            y2: Bottom-right y-coordinate.
        """
        return cls(x1=x1, y1=y1, w=x2 - x1, h=y2 - y1)
    
    def to_xyxy(self) -> tuple[int, int, int, int]:
        """
        Converts the bounding box to top-left (x1, y1) and bottom-right (x2, y2) coordinates.

        Returns:
            A tuple containing (x1, y1, x2, y2).
        """
        x2 = self.x1 + self.w
        y2 = self.y1 + self.h
        return (int(self.x1), int(self.y1), int(x2), int(y2))


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

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
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
            ret, frame_bgr = self.cap.read()     # Read frame in BGR format
            if not ret:
                logger.info("Reached end of video or error reading frame.")
                break # Exit the loop if no frame is returned
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Converts to RGB
            yield frame_rgb

    def get_frame_by_index(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Retrieves a specific frame from the video by its index.

        Args:
            frame_index: The 0-based index of the frame to retrieve.

        Returns:
            The frame as a numpy array in RGB format, or None if the index is
            out of bounds or an error occurs.
        """
        if not self.cap or not self.cap.isOpened():
            logger.error("Video capture is not initialized or opened for get_frame_by_index.")
            return None
        if self.in_n_frames is None or not (0 <= frame_index < self.in_n_frames):
            if self.in_n_frames is not None:
                logger.error(f"Frame index {frame_index} is out of bounds (0-{self.in_n_frames - 1}).")
            else:
                logger.error(f"Frame index {frame_index} is out of bounds (total frames unknown).")
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame_bgr = self.cap.read() # Read frame in BGR format
        if not ret:
            logger.error(f"Failed to read frame at index {frame_index}.")
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB

    def draw_detections(self, frame_bgr: np.ndarray, detections: list) -> np.ndarray:
        """
        Draws bounding boxes and labels on the frame for detected objects.

        Args:
            frame: The input video frame as a numpy array. This frame will be modified in place.
            detections: A list of detected objects with their bounding boxes and scores.

                        Expected format for each item in detections:
                         ([x1, y1, w, h], confidence, class_id)
                         where [x1, y1, w, h] are the coordinates for the top-left corner
                         and width/height of the bounding box.

        Returns:
            The input frame, modified with drawn detections.
        """
        for bbox_coords, confidence, class_id in detections:
            if not (isinstance(bbox_coords, (list, tuple)) and len(bbox_coords) == 4):
                logger.warning(f"Unexpected bbox_coords format in draw_detections: {bbox_coords}. Skipping.")
                continue
            # bbox_coords is [x1, y1, w, h]
            x1, y1, w, h = map(int, bbox_coords)
            # Calculate bottom-right coordinates for cv2.rectangle
            x2, y2 = x1 + w, y1 + h # These are coordinates, not image data
            
            # Define the two lines of text
            # label_line1 = f"Cls {class_id}" # Removed first line
            label_line2 = f"{confidence*100:.0f}%" # Format as percentage
            
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4 
            font_thickness = 1
            text_color_bgr = (0, 255, 0) # Green
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Get text size for the (now single) line to position it
            (text_width_line2, text_height_line2), baseline_line2 = cv2.getTextSize(label_line2, font_face, font_scale, font_thickness)
            
            # Position for the single line (above the box)
            text_y_pos = y1 - 5 # Position 5 pixels above the top of the box
            if text_y_pos < text_height_line2 : # Ensure it's not drawn off the top of the image
                text_y_pos = y1 + text_height_line2 + 5 # If too high, draw inside and below top of box
            
            cv2.putText(frame_bgr, label_line2, (x1, text_y_pos), font_face, font_scale, text_color_bgr, font_thickness)
            
        return frame_bgr

    def draw_tracks(self, frame: np.ndarray, tracks: list,
                    team_id_getter: Optional[Callable[[BoundingBox, np.ndarray, str], Optional[int]]] = None) -> np.ndarray:
        """
        Draws bounding boxes and track IDs on the frame for tracked objects.
        Colors tracks based on team ID if team_id_getter is provided.

        Args:
            frame: The input video frame as a numpy array (expected in RGB).
            tracks: A list of Track objects from deep_sort_realtime.
            team_id_getter: An optional function that takes a BoundingBox, the current frame,
                            and the track_id, and returns a team ID.

        Returns:
            The modified frame (RGB) with drawn track information.
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
                original_detection_bbox = BoundingBox(x1=original_ltwh[0],
                                                      y1=original_ltwh[1],
                                                      w=original_ltwh[2],
                                                      h=original_ltwh[3])
                team_id = team_id_getter(original_detection_bbox, frame, track_id) # Pass track_id, getter expects RGB frame

            # Determine color based on team_id
            if team_id == 0:
                track_color = (0, 0, 255)   # Red for team 0 (RGB)
            elif team_id == 1:
                track_color = (255, 0, 0) # Blue for team 1 (RGB)
            else: # Default color if no team or team_id_getter not provided
                continue
                 # track_color = (0, 255, 0)   # Green for unknown/unassigned (RGB)
            
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), track_color, 2)
            label = f"ID: {track_id}"
            if team_id is not None:
                label += f" T: {team_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2)
        return frame
