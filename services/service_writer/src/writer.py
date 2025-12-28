import logging
logger = logging.getLogger(__name__)

import colorsys
import random
from typing import List, Tuple

import cv2

from shared_libs.common import track_serialization

import json
import os

class Writer:
    def __init__(self, video_path: str, tracks_path: str, output_path: str, players_path: str = None):
        self.video_path = video_path
        self.tracks_by_frame = track_serialization.load_for_writing(tracks_path)
        self.output_path = output_path
        self.player_mapping = {}
        
        if players_path and os.path.exists(players_path):
            try:
                with open(players_path, 'r') as f:
                    player_data = json.load(f)
                    # Mapping is stored as strings in JSON but we want ints
                    self.player_mapping = {
                        int(tid): val 
                        for tid, val in player_data.get('track_to_player', {}).items()
                    }
                logger.info(f"Loaded {len(self.player_mapping)} player assignments from {players_path}")
            except Exception as e:
                logger.error(f"Failed to load players.json: {e}")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("Error opening video file")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        output = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
                
            frame_tracks = self.tracks_by_frame.get(frame_count, [])  
            for track in frame_tracks:
                track_id = track["track_id"]
                
                # Default values
                display_id = str(track_id)
                color = get_color(track_id)
                
                # Override if player assignment exists
                if track_id in self.player_mapping:
                    mapping = self.player_mapping[track_id]
                    display_id = f"P{mapping['player_id']}"
                    team_id = mapping.get('team_id', 0)
                    color = get_team_color(team_id)
                
                bbox = track["bbox"]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, display_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            output.write(frame_bgr)
            frame_count += 1
            if frame_count % 100 == 0:
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


def get_team_color(team_id: int) -> Tuple[int, int, int]:
    """
    Returns a color based on the team_id.
    """
    team_colors = {
        0: (0, 0, 255),    # Red
        1: (255, 0, 0),    # Blue
        2: (0, 255, 255),  # Yellow
        -1: (0, 255, 0)    # Green
    }
    return team_colors.get(team_id, (255, 255, 255)) # Default White


