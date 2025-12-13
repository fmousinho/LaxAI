





import logging
logger = logging.getLogger(__name__)

import colorsys
import random
from typing import List, Tuple



import json
import cv2
import colorsys
import random
from typing import List, Tuple

from shared_libs.common import track_serialization

class Writer:
    def __init__(self, video_path: str, tracks_path: str, output_path: str):
        self.video_path = video_path
        self.tracks_by_frame = track_serialization.load_for_writing(tracks_path)
        self.output_path = output_path


    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("Error opening video file")
            return
        output = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        frame_count = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_tracks = self.tracks_by_frame.get(frame_count, [])  
            for track in frame_tracks:
                cv2.rectangle(frame_bgr, tuple(track["bbox"][0:2]), tuple(track["bbox"][2:4]), get_color(track["track_id"]), 2)
                cv2.putText(frame_bgr, str(track["track_id"]), tuple(track["bbox"][0:2]), cv2.FONT_HERSHEY_SIMPLEX, .6, get_color(track["track_id"]), 2)
            output.write(frame_bgr)
            frame_count += 1
            if frame_count % 50 == 0:
                logger.info(f"Processed frame {frame_count}")
        cap.release()
        output.release()
    




def _generate_distinct_colors(n: int = 50) -> List[Tuple[int, int, int]]:
    """
    Generates N distinct colors in BGR format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.9  # High saturation for vivid colors
        value = 0.9       # High value for brightness
        
        # colorsys returns RGB in [0, 1]
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to 0-255 and BGR (for OpenCV compatibility)
        b_int = int(b * 255)
        g_int = int(g * 255)
        r_int = int(r * 255)
        
        colors.append((b_int, g_int, r_int))
    
    # Shuffle to ensure adjacent IDs don't always have adjacent hue
    # (Optional, but often looks better. If we want a smooth gradient, remove shuffle)
    # Using a fixed seed for reproducibility across runs
    random.seed(42) 
    random.shuffle(colors)
    
    return colors

# Pre-compute the colors
_COLORS = _generate_distinct_colors(50)
        

def get_color(track_id: int) -> Tuple[int, int, int]:
    """
    Returns one of 50 distinct colors based on the track_id.
    If the track_id is larger than 50, it rotates through the colors.
    
    Args:
        track_id: The integer ID of the track.
        
    Returns:
        A tuple of (B, G, R) integers.
    """
    return _COLORS[track_id % len(_COLORS)]


