import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import supervision as sv

from core.common.google_storage import get_storage
from core.common.detection import DetectionModel

logger = logging.getLogger(__name__)


class TrainPipeline:
    """
    Training pipeline that processes MP4 videos from Google Storage.
    
    This pipeline:
    1. Creates a processing run folder with GUID
    2. Processes videos from the raw directory
    3. Validates video resolution (minimum 1920x1080)
    4. Extracts frames with sufficient detections
    5. Saves metadata and detection results
    """
    
    def __init__(self, user_path: str):
        """
        Initialize the training pipeline.
        
        Args:
            user_path: The user path in Google Storage (e.g., "tenant1/user")
        """
        self.user_path = user_path
        self.storage_client = get_storage(user_path)
        self.detection_model = DetectionModel()
        self.run_guid = str(uuid.uuid4())
        self.run_folder = f"process/run_{self.run_guid}"
        
        logger.info(f"Initialized TrainPipeline for user: {user_path}")
        logger.info(f"Run GUID: {self.run_guid}")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary with pipeline results and statistics
        """
        logger.info("Starting training pipeline execution")
        
        # Create run folder
        self._create_run_folder()
        
        # Get all MP4 videos from raw directory
        videos = self._get_raw_videos()
        
        if not videos:
            logger.warning("No MP4 videos found in raw directory")
            return {"status": "completed", "videos_processed": 0, "errors": []}
        
        # Process each video
        results = {
            "status": "completed",
            "run_guid": self.run_guid,
            "run_folder": self.run_folder,
            "videos_processed": 0,
            "videos_skipped": 0,
            "errors": []
        }
        
        for video_blob_name in videos:
            try:
                video_result = self._process_video(video_blob_name)
                if video_result["success"]:
                    results["videos_processed"] += 1
                else:
                    results["videos_skipped"] += 1
                    results["errors"].append(video_result["error"])
            except Exception as e:
                logger.error(f"Error processing video {video_blob_name}: {e}")
                results["videos_skipped"] += 1
                results["errors"].append(f"Video {video_blob_name}: {str(e)}")
        
        logger.info(f"Pipeline completed. Processed: {results['videos_processed']}, Skipped: {results['videos_skipped']}")
        return results
    
    def _create_run_folder(self) -> None:
        """Create the run folder in Google Storage."""
        # Create a placeholder file to ensure the folder exists
        placeholder_content = json.dumps({
            "run_guid": self.run_guid,
            "created_at": datetime.now().isoformat(),
            "user_path": self.user_path
        })
        
        placeholder_blob = f"{self.run_folder}/.run_info.json"
        
        if not self.storage_client.upload_from_string(placeholder_blob, placeholder_content):
            raise RuntimeError(f"Failed to create run folder: {self.run_folder}")
        
        logger.info(f"Created run folder: {self.run_folder}")
    
    def _get_raw_videos(self) -> List[str]:
        """
        Get all MP4 videos from the raw directory.
        
        Returns:
            List of video blob names
        """
        raw_prefix = "raw/"
        blobs = self.storage_client.list_blobs(prefix=raw_prefix)
        
        videos = []
        for blob_name in blobs:
            if blob_name.lower().endswith('.mp4'):
                videos.append(blob_name)
        
        logger.info(f"Found {len(videos)} MP4 videos in raw directory")
        return videos
    
    def _process_video(self, video_blob_name: str) -> Dict[str, Any]:
        """
        Process a single video through the pipeline.
        
        Args:
            video_blob_name: The blob name of the video in Google Storage
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video: {video_blob_name}")
        
        try:
            # Generate video GUID and folder
            video_guid = str(uuid.uuid4())
            video_folder = f"{self.run_folder}/video_{video_guid}"
            
            # Download video temporarily for processing
            temp_video_path = f"/tmp/video_{video_guid}.mp4"
            
            if not self.storage_client.download_blob(video_blob_name, temp_video_path):
                return {"success": False, "error": f"Failed to download video: {video_blob_name}"}
            
            # Validate video resolution
            if not self._validate_video_resolution(temp_video_path):
                os.remove(temp_video_path)
                return {"success": False, "error": f"Video resolution too low: {video_blob_name}"}
            
            # Extract video metadata
            metadata = self._extract_video_metadata(temp_video_path, video_blob_name, video_guid)
            
            # Extract frames with detections
            frames_data = self._extract_frames_with_detections(temp_video_path)
            
            if len(frames_data) < 20:
                os.remove(temp_video_path)
                return {"success": False, "error": f"Could not extract 20 frames with sufficient detections: {video_blob_name}"}
            
            # Move video to its folder in Google Storage
            new_video_name = f"video_{video_guid}.mp4"
            new_video_blob = f"{video_folder}/{new_video_name}"
            
            if not self.storage_client.upload_from_file(new_video_blob, temp_video_path):
                os.remove(temp_video_path)
                return {"success": False, "error": f"Failed to upload processed video: {video_blob_name}"}
            
            # Delete original video from raw folder
            if not self.storage_client.delete_blob(video_blob_name):
                logger.warning(f"Failed to delete original video: {video_blob_name}")
            
            # Save metadata
            metadata_blob = f"{video_folder}/metadata.json"
            metadata_content = json.dumps(metadata, indent=2)
            if not self.storage_client.upload_from_string(metadata_blob, metadata_content):
                logger.warning(f"Failed to save metadata for video: {video_blob_name}")
            
            # Save detections
            detections_blob = f"{video_folder}/detections.json"
            detections_content = json.dumps(frames_data, indent=2)
            if not self.storage_client.upload_from_string(detections_blob, detections_content):
                logger.warning(f"Failed to save detections for video: {video_blob_name}")
            
            # Clean up temporary file
            os.remove(temp_video_path)
            
            logger.info(f"Successfully processed video: {video_blob_name} -> {video_folder}")
            return {
                "success": True,
                "video_guid": video_guid,
                "video_folder": video_folder,
                "frames_extracted": len(frames_data)
            }
            
        except Exception as e:
            # Clean up temporary file if it exists
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            raise e
    
    def _validate_video_resolution(self, video_path: str) -> bool:
        """
        Validate that video resolution is at least 1920x1080.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if resolution is adequate, False otherwise
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        is_valid = width >= 1920 and height >= 1080
        logger.info(f"Video resolution: {width}x{height}, Valid: {is_valid}")
        
        return is_valid
    
    def _extract_video_metadata(self, video_path: str, original_blob_name: str, video_guid: str) -> Dict[str, Any]:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to the video file
            original_blob_name: Original blob name in Google Storage
            video_guid: Generated GUID for the video
            
        Returns:
            Dictionary with video metadata
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for metadata extraction: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        metadata = {
            "video_guid": video_guid,
            "original_name": os.path.basename(original_blob_name),
            "original_blob_name": original_blob_name,
            "processed_name": f"video_{video_guid}.mp4",
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "processed_at": datetime.now().isoformat()
        }
        
        return metadata
    
    def _extract_frames_with_detections(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract 20 frames with at least 10 detections each.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of frame data with detections
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for frame extraction: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate initial frame positions (20 frames equally spaced)
        frame_positions = [int(i * total_frames / 20) for i in range(20)]
        
        frames_data = []
        
        for target_frame in frame_positions:
            frame_data = self._find_frame_with_detections(cap, target_frame, total_frames)
            if frame_data:
                frames_data.append(frame_data)
        
        cap.release()
        
        logger.info(f"Extracted {len(frames_data)} frames with sufficient detections")
        return frames_data
    
    def _find_frame_with_detections(self, cap: cv2.VideoCapture, start_frame: int, total_frames: int) -> Optional[Dict[str, Any]]:
        """
        Find a frame with at least 10 detections, searching up to 5 frames ahead.
        
        Args:
            cap: OpenCV video capture object
            start_frame: Starting frame position
            total_frames: Total frames in video
            
        Returns:
            Frame data with detections, or None if not found
        """
        for offset in range(6):  # Search current frame + 5 frames ahead
            frame_pos = start_frame + offset
            
            if frame_pos >= total_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Run detection on frame
            detections = self.detection_model.generate_detections(frame)
            
            if len(detections) >= 10:
                # Convert detections to serializable format
                detections_data = {
                    "frame_position": frame_pos,
                    "detections_count": len(detections),
                    "detections": {
                        "xyxy": detections.xyxy.tolist(),
                        "confidence": detections.confidence.tolist() if detections.confidence is not None else None,
                        "class_id": detections.class_id.tolist() if detections.class_id is not None else None
                    }
                }
                
                logger.debug(f"Found {len(detections)} detections at frame {frame_pos}")
                return detections_data
        
        logger.warning(f"Could not find frame with 10+ detections starting from frame {start_frame}")
        return None


def run_training_pipeline(user_path: str) -> Dict[str, Any]:
    """
    Convenience function to run the training pipeline.
    
    Args:
        user_path: The user path in Google Storage (e.g., "tenant1/user")
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline = TrainPipeline(user_path)
    return pipeline.run_pipeline()


if __name__ == "__main__":
    # Example usage
    user_path = "tenant1/user"
    results = run_training_pipeline(user_path)
    
    print("Pipeline Results:")
    print(json.dumps(results, indent=2))

