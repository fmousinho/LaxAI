
from typing_extensions import Tuple, List
import numpy as np
import shared_libs.config.logging_config
from supervision import Detections


def draw_players (frame: np.ndarray, detections: Detections, frame_index: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Draw players on the given video frame based on detections.

    Args:
        frame (np.ndarray): The video frame to draw on.
        detections (Detections): The detections containing player information.
        frame_index (int): The index of the current frame.

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]]]: The frame with players drawn and a list of player positions.
    """
    players_positions = []

    # Filter detections for the current frame
    mask = detections.data["frame_index"] == frame_index
    frame_detections = detections.data[mask]

    for det in frame_detections:
        if det["class"] == "player":
            bbox = det["bbox"]  # Assuming bbox is in [x1, y1, x2, y2] format
            x1, y1, x2, y2 = map(int, bbox)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Calculate center position
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            players_positions.append((center_x, center_y))
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

    return frame, players_positions