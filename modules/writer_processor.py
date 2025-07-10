import logging
import cv2
import supervision as sv
import numpy as np
from tqdm import tqdm
from collections import deque
from typing import Dict, List, Optional, Tuple
from modules.player import Player

logger = logging.getLogger(__name__)


class VideoWriterProcessor:
    """
    Processor for writing annotated video frames with player tracking information.
    """
    
    def __init__(self, 
                 output_video_path: str,
                 video_info: sv.VideoInfo,
                 team_colors: Dict[int, Tuple[int, int, int]]):
        """
        Initialize the video writer processor.
        
        Args:
            output_video_path: Path where the processed video will be saved
            video_info: Video information object from supervision
            team_colors: Dictionary mapping team IDs to RGB color tuples
        """
        self.output_video_path = output_video_path
        self.video_info = video_info
        self.team_colors = team_colors
        
        # Initialize supervision annotators
        self.ellipse_annotator = sv.EllipseAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def write_annotated_video(
        self,
        frames_generator,
        multi_frame_detections: list,
        track_to_player: Dict[int, Player],
        frame_target: int):
        """
        Write annotated video frames with player tracking information.
        
        Args:
            frames_generator: Generator yielding video frames
            multi_frame_detections: List containing detection data for each frame
            track_to_player: Mapping from tracker ID to Player object
            frame_target: Total number of frames to process
        """
        logger.info("Creating output video with annotated frames")

        multi_frame_detections = deque(multi_frame_detections)
        
        with sv.VideoSink(target_path=self.output_video_path, video_info=self.video_info) as sink:
            for frame in tqdm(frames_generator, desc="Writing frames", total=frame_target):
                try:
                    detections: sv.Detections = multi_frame_detections.popleft()
                except IndexError:
                    logger.error("Detections could not be retrieved to write a frame.")
                    break
                    
                if detections.tracker_id is None or detections.confidence is None:
                    logger.error("Missing tracking ID or confidence in detections.")
                    continue

                # Annotate the frame
                annotated_frame = self._annotate_frame(frame, detections, track_to_player)
                
                # Write the annotated frame
                sink.write_frame(frame=annotated_frame)
                
        logger.info(f"Video writing complete. Output saved to: {self.output_video_path}")
        
        def _annotate_frame(self, 
                        frame: np.ndarray, 
                        detections: sv.Detections, 
                        track_to_player: Dict[int, Player]) -> np.ndarray:
            """
            Annotate a single frame with player tracking information.
            
            Args:
                frame: The input frame to annotate
                detections: Detection data for this frame
                track_to_player: Mapping from tracker ID to Player object
                
            Returns:
                Annotated frame with ellipses and labels
            """
            # Check if there are any detections to annotate
            if len(detections) == 0:
                return frame.copy()
                
            # Prepare detections data for supervision annotators
            detections.data["player_id"] = np.zeros(len(detections), dtype=np.int16)
            detections.data["team_id"] = np.full(len(detections), -1, dtype=np.int16)

            # Prepare lists for labels and colors
            labels = []
            colors = []

            # Loop through all detections in a given frame, and assign a player_id to all confirmed ones
            for i in range(len(detections)):
                tracker_id_np = detections.tracker_id[i]
                if tracker_id_np is None:
                    detections.data["player_id"][i] = -1
                    detections.data["team_id"][i] = -1
                    labels.append("?")
                    colors.append((64, 64, 64))  # Dark gray for untracked
                    continue
                    
                tid = int(tracker_id_np)
                player = track_to_player.get(tid, None)
                
                if player is None:
                    logger.warning(f"Tracker ID {tid} has no associated Player object. Skipping annotation.")
                    detections.data["player_id"][i] = -1
                    detections.data["team_id"][i] = -1
                    labels.append("?")
                    colors.append((128, 128, 128))  # Gray for unknown
                else:
                    detections.data["player_id"][i] = player.id
                    team = player.team if player.team is not None else -1
                    detections.data["team_id"][i] = team
                    
                    # Use player ID as label
                    labels.append(f"P{player.id:02d}")
                    
                    # Get team-based color
                    color = self.team_colors.get(team, (128, 128, 128))  # Default to gray
                    colors.append(color)

            # Create annotated frame using supervision annotators
            annotated_frame = frame.copy()
            
            # Try different approaches for color annotation
            try:

                detections.data["colors"] = np.array(colors, dtype=np.uint8)
                
                # Use supervision ellipse annotator
                annotated_frame = self.ellipse_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections
                )
                
                # Use supervision label annotator for player IDs
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )
            except Exception as e:
                logger.error(f"Error annotating frame:")
                logger.error(e)
            
            return annotated_frame
        
       
