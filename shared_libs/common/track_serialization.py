import json
import os
import numpy as np
from typing import List, Union, Dict, Any

def save(
    objects_per_frame: List[List[Any]],
    video_source: str,
    save_path: str
):
    """
    Save tracks or detections to a JSON file.

    Args:
        objects_per_frame: List of frames, where each frame contains a list of objects.
                           Objects can be STrack instances or numpy arrays [x1, y1, x2, y2, score, track_id].
        video_source: Path to the source video.
        save_path: Output JSON path.
        use_track_id: Whether to look for and save track_id.
    """
    serialized_frames = []
    
    for frame_idx, frame_objects in enumerate(objects_per_frame):
        track_objects = []
        for obj in frame_objects:
            # Handle STrack objects
            if hasattr(obj, 'tlbr'):
                bbox = [int(x) for x in obj.tlbr]
                score = float(obj.score)
                track_id = int(obj.track_id)
            # Handle numpy arrays [x1, y1, x2, y2, score] or [x1, y1, x2, y2, score, track_id]
            elif isinstance(obj, np.ndarray):
                bbox = [int(x) for x in obj[:4]]
                score = float(obj[4]) if len(obj) > 4 else 0.0
                if len(obj) > 5:
                     track_id = int(obj[5])
                else:
                     track_id = -1
            else:
                logger.error(f"Unsupported object type: {type(obj)}")
                return

            track_dict = {
                "track_id": track_id,
                "bbox": bbox,
                "score": score
            }
            track_objects.append(track_dict)
            
        serialized_frames.append({
            "frame_id": frame_idx, # 0-indexed
            "track_objects":  
        })

    output_data = {
        "video_source": video_source,
        "frames": serialized_frames
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(output_data, f, indent=4)

def load_for_writing(json_path: str) -> Dict[int, List[Dict]]:
    """
    Load tracks/detections from JSON for modification/writing.
    Returns a dictionary keyed by frame_id (int) -> list of track objects.
    """
    if not os.path.exists(json_path):
        return {}
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    frames_dict = {}
    for frame_data in data.get("frames", []):
        frame_id = int(frame_data.get("frame_id", -1))
        if frame_id >= 0:
            frames_dict[frame_id] = frame_data.get("track_objects", [])
            
    return frames_dict
