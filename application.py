
import logging
from typing import Optional
import torch
import supervision as sv
from tqdm import tqdm

from .modules.detection import DetectionModel
from .tools.store_driver import Store

logger = logging.getLogger(__name__)


def run_application (
        store: Store,
        input_video: str,
        output_video_path: str = "results.mp4",
        device: torch.device = torch.device("cpu"),
        debug_max_frames: Optional[int] = None
    ):
    """
    Main entry point for the video processing application.
    Args:
        store: The Store instance for file management.
        input_video: Path to the input video file.
        output_video_path: Path where the processed video will be saved.
        device: The torch device to use for processing (e.g., "cpu", "cuda").
        debug_max_frames: If set, limits processing to this many frames for debugging.
    """

    video_info = sv.VideoInfo.from_video_path(video_path=input_video) #type: ignore
    frames_generator = sv.get_video_frames_generator(source_path=input_video) #type: ignore
    model = DetectionModel(store=store, device=device)

    tracker = sv.ByteTrack(     #type: ignore
        track_activation_threshold = 0.25,
        lost_track_buffer = 30,
        minimum_matching_threshold=0.95,
        frame_rate = video_info.fps,
        minimum_consecutive_frames=1
        ) 
    smoother = sv.DetectionsSmoother() #type: ignore

    ellipse_annotator = sv.EllipseAnnotator() #type: ignore
    label_annotator = sv.LabelAnnotator() #type: ignore

    frame_id = 0
    frame_target = debug_max_frames if debug_max_frames else video_info.total_frames
    with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink: #type: ignore
        for frame in tqdm(frames_generator, desc="Processing frames", total=frame_target):
            detections = model.generate_detections(frame)
            detections = tracker.update_with_detections(detections)
            # detections = smoother.update_with_detections(detections)

            annotated_frame = ellipse_annotator.annotate(scene=frame.copy(), detections=detections)

            if detections.tracker_id is not None and detections.confidence is not None:
                labels = [
                    f"{tracker_id} {confidence:.2f}" 
                    for tracker_id, confidence in zip(detections.tracker_id, detections.confidence)
                ]
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
                )


            sink.write_frame(frame=annotated_frame)
            frame_id += 1
            if debug_max_frames and frame_id >= debug_max_frames:
                break


