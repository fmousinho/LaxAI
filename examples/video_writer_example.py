# Example usage of the VideoWriterProcessor

"""
Example showing how to use the VideoWriterProcessor independently
for writing annotated video frames.
"""

import supervision as sv
import torch
from collections import deque
from modules.writer_processor import VideoWriterProcessor
from modules.player import Player

# Example team colors
TEAM_COLORS = {
    0: (255, 0, 0),      # Red for team 0
    1: (0, 0, 255),      # Blue for team 1
    2: (0, 255, 0),      # Green for Referees
    10: (255, 255, 0),   # Yellow for Goalies team 0
    11: (255, 165, 0),   # Orange for Goalies team 1
    -1: (128, 128, 128)  # Gray for unknown
}

def example_video_writing():
    """Example of using VideoWriterProcessor to write annotated videos."""
    
    # Input video information
    input_video = "path/to/input/video.mp4"
    output_video = "path/to/output/annotated_video.mp4"
    
    # Get video info
    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    
    # Initialize the writer processor
    writer_processor = VideoWriterProcessor(
        output_video_path=output_video,
        video_info=video_info,
        team_colors=TEAM_COLORS
    )
    
    # Example: Create some mock data (in practice, these would come from your detection/tracking pipeline)
    frames_generator = sv.get_video_frames_generator(source_path=input_video)
    
    # Mock detections for each frame (in practice, these would come from your detection processor)
    multi_frame_detections = deque()
    # ... populate with actual detection data
    
    # Mock player mapping (in practice, this would come from your player association)
    track_to_player = {
        1: Player(id=1, team=0),  # Player 1, Team 0 (Red)
        2: Player(id=2, team=1),  # Player 2, Team 1 (Blue)
        3: Player(id=3, team=0),  # Player 3, Team 0 (Red)
        # ... more players
    }
    
    # Write the annotated video
    writer_processor.write_annotated_video(
        frames_generator=frames_generator,
        multi_frame_detections=multi_frame_detections,
        track_to_player=track_to_player,
        frame_target=video_info.total_frames
    )
    
    print(f"Annotated video saved to: {output_video}")

def example_custom_annotation():
    """Example of customizing the VideoWriterProcessor for different annotation styles."""
    
    # Custom team colors
    custom_colors = {
        0: (0, 255, 255),    # Cyan for team 0
        1: (255, 0, 255),    # Magenta for team 1
        2: (255, 255, 0),    # Yellow for referees
        -1: (64, 64, 64)     # Dark gray for unknown
    }
    
    input_video = "path/to/input/video.mp4"
    output_video = "path/to/output/custom_annotated_video.mp4"
    
    video_info = sv.VideoInfo.from_video_path(video_path=input_video)
    
    # Initialize with custom colors
    writer_processor = VideoWriterProcessor(
        output_video_path=output_video,
        video_info=video_info,
        team_colors=custom_colors
    )
    
    # The rest of the usage remains the same
    # ... (setup frames_generator, multi_frame_detections, track_to_player)
    
    print("Custom annotation example setup complete")

if __name__ == "__main__":
    print("=== VideoWriterProcessor Usage Examples ===")
    print("Note: Update the file paths in the examples before running")
    
    # example_video_writing()
    # example_custom_annotation()
    
    print("Examples ready to run (uncomment the function calls above)")
