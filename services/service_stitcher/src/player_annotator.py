import numpy as np
import shared_libs.config.logging_config

import supervision as sv
from shared_libs.common.player_color_palette import player_color_palette

def annotate_with_players(
    frame: np.ndarray,
    detections: sv.Detections,
    player_color_palette: sv.ColorPalette = player_color_palette
) -> np.ndarray:
    """
    Annotate the given frame with tracker IDs and bounding boxes.

    Args:
        frame (np.ndarray): The input video frame to annotate.
        detections (sv.Detections): Detections object containing bounding boxes and tracker IDs.
        player_color_palette (sv.ColorPalette): Color palette for different players.

    Returns:
        np.ndarray: Annotated frame with bounding boxes and tracker ID labels.
    """
    
    annotator_params = {
        "color": player_color_palette,
        "color_lookup": sv.ColorLookup.TRACK
    }

    box_annotator = sv.BoxAnnotator(**annotator_params)
    label_annotator = sv.LabelAnnotator(**annotator_params)

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections)
    
    return annotated_frame