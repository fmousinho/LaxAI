import os
import json
import time
import random
import threading
import typing
import logging
import concurrent.futures
import io
import numpy as np
from typing import Dict, Tuple, cast, Any
from PIL import Image
from supervision import Detections

from frame_cache import RollingFrameCache

import shared_libs.config.logging_config 
from shared_libs.common.player_manager import initialize_player_manager, load_player_manager
from shared_libs.common.google_storage import get_storage, GCSPaths
from shared_libs.common.detection_utils import json_to_detections, detections_to_json
from shared_libs.common.rendering_config import RenderingConfig
from shared_libs.common.detection_utils import create_frame_response

logger = logging.getLogger(__name__)

FRAME_SKIP_INTERVAL = 30  # Default frame skip interval


class VideoManager:
    """Manages video playback sessions with Google Cloud Storage integration.
    
    This class provides functionality to load videos from GCS and manage
    frame-by-frame playback sessions. Each instance represents a single
    video session with a unique session ID for tracking purposes.
    
    Attributes:
        lock (threading.Lock): Thread synchronization lock for session management
        session_id (str): Unique identifier for this video session
        storage (GoogleStorageClient): GCS client for video access
        cap (GCSVideoCapture): Video capture object for frame access
        current_frame_id (int): Current frame position in the video
        total_frames (int): Total number of frames in the loaded video
        frame_skip_interval (int): Number of frames to skip between reads for performance
    """
    path_manager = GCSPaths()

    def __init__(self, tenant_id: str, frame_skip_interval: int = FRAME_SKIP_INTERVAL):
        """Initialize a new VideoManager instance.
        
        Args:
            tenant_id (str): The tenant identifier for GCS access
            frame_skip_interval (int): Number of frames to skip between reads (default: 30)
            
        Raises:
            ValueError: If storage backend initialization fails
        """
        self.lock = threading.Lock()
        self.session_id = None
        self.session_id = self._generate_session_id()
        self.storage = get_storage(tenant_id=tenant_id)
        if not self.storage or not self.session_id: 
            raise ValueError("Failed to initialize storage backend.")
        self.cap = None
        self.current_frame_id = None
        self.total_frames = 0
        self.frame_skip_interval = frame_skip_interval
        self.video_id = None
        self.detections: Detections = Detections.empty()
        self.player_manager = None
        
        # Initialize rolling frame cache (5 frames before/after)
        self.frame_cache = RollingFrameCache(
            window_size=5,
            max_workers=2,
            format="jpeg",
            quality=85,
            frame_skip_interval=self.frame_skip_interval
        )

    def __del__(self):
        """Cleanup method to properly close video capture when VideoManager is destroyed."""
        if self.frame_cache:
            self.frame_cache.shutdown()
        if self.cap:
            try:
                self.cap.__exit__(None, None, None)
            except Exception:
                pass  # Ignore cleanup errors during destruction
            self.cap = None


    def load_video(self, video_path: str) -> dict:
        """Load a video from Google Cloud Storage for playback.
        
        Initializes the video capture object and retrieves video metadata.
        Sets up the session for frame-by-frame access.
        
        Args:
            video_path (str): GCS path to the video file
            
        Returns:
            dict: Video loading result containing:
                - success (bool): Whether loading was successful
                - session_id (str): Unique session identifier
                - video_path (str): The video path that was loaded
                - total_frames (int): Total number of frames in the video
                - has_next_frame (bool): Whether there are more frames to read
                - has_previous_frame (bool): Whether there are previous frames
                
        Raises:
            ValueError: If video file is not found or cannot be loaded
        """
        try:
            # Extract video_id from path: tenant/process/{video_id}/imported/filename.mp4
            path_parts = video_path.split("/")
            if len(path_parts) >= 4 and path_parts[1] == "process":
                self.video_id = path_parts[2]
            else:
                # Fallback to basename if path doesn't match expected structure
                self.video_id = os.path.basename(video_path).split(".")[0]

            # Run detections loading and video capture setup in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                detections_future = executor.submit(self._load_detections)
                capture_future = executor.submit(self._setup_video_capture, video_path)
                players_future = executor.submit(self._load_players_if_json_exists)
                
                # Wait for both to complete
                detections_result = detections_future.result()
                capture_result = capture_future.result()
                players_result = players_future.result()
            # Check if detections loading succeeded
            if not detections_result:
                # Clean up video capture if detections failed
                if "cap" in capture_result:
                    capture_result["cap"].__exit__(None, None, None)
                raise ValueError("Detections could not be loaded for the video.")
            
            # Set up video capture properties
            self.cap = capture_result["cap"]
            self.total_frames = capture_result["total_frames"]
            
            result = {
                "session_id": self.session_id,
                "total_frames": self.total_frames,
                "has_next_frame": self.total_frames > self.frame_skip_interval,
                "has_previous_frame": False,
            }

        except FileNotFoundError:
            raise ValueError(f"Video file not found at path: {video_path}")
        except Exception as e:
            raise ValueError(f"Error loading video: {e}")

        return result
    

    def next_frame(self) -> dict:
        """Navigate to next frame and return metadata only.
        
        Returns:
            dict: Frame metadata for client-side annotation
                
        Raises:
            ValueError: If no video is loaded or end of video reached
        """
        if self.current_frame_id is None:
            new_frame_id = 0
        else:
            new_frame_id = self.current_frame_id + self.frame_skip_interval
            if new_frame_id >= self.total_frames:
                raise ValueError("End of video reached")
        
        self.current_frame_id = new_frame_id
        return self.get_frame_metadata(new_frame_id)

    def previous_frame(self) -> dict:
        """Navigate to previous frame and return metadata only.
        
        Returns:
            dict: Frame metadata for client-side annotation
                
        Raises:
            ValueError: If no video is loaded
        """
        if self.current_frame_id is None:
            new_frame_id = 0
        else:
            new_frame_id = max(0, self.current_frame_id - self.frame_skip_interval)
        
        self.current_frame_id = new_frame_id
        return self.get_frame_metadata(new_frame_id)

    # ========== Private Helper Methods ==========
    
    def _load_raw_frame(self, frame_id: int) -> np.ndarray:
        """Load raw RGB frame from video without processing.
        
        Args:
            frame_id: Frame index to load
            
        Returns:
            np.ndarray: Raw RGB frame data
            
        Raises:
            ValueError: If frame cannot be loaded
        """
        if not self.cap:
            raise ValueError("No video loaded for this session")
        
        if not self.cap.set("CAP_PROP_POS_FRAMES", frame_id):
            raise ValueError(f"Failed to seek to frame {frame_id}")
        
        success, frame_rgb = self.cap.read(return_format="rgb")
        if not success or frame_rgb is None:
            raise ValueError(f"Failed to read frame {frame_id}")
        
        return frame_rgb
    
    def _get_frame_detections(self, frame_id: int) -> Detections:
        """Get detections for specific frame from pre-loaded data.
        
        Args:
            frame_id: Frame index
            
        Returns:
            Detections: Detection objects for the frame
            
        Raises:
            ValueError: If detection extraction fails
        """
        mask = self.detections.data["frame_index"] == frame_id
        frame_detections = self.detections[mask]
        
        if not isinstance(frame_detections, Detections):
            raise ValueError("Detection extraction failed - not a Detections object")
        
        return frame_detections
    
    def _ensure_player_manager(self, frame_id: int, detections: Detections) -> None:
        """Initialize player_manager if not already initialized.
        
        Args:
            frame_id: Current frame index
            detections: Detections for initialization
        """
        if not self.player_manager and self.video_id is not None:
            self.player_manager = initialize_player_manager(
                self.video_id, 
                frame_id, 
                detections
            )
    
    def _apply_player_mapping(self, detections: Detections) -> Detections:
        """Map tracker_id to player_id in detections.
        
        Args:
            detections: Detections with tracker_ids
            
        Returns:
            Detections: Detections with player_ids mapped
        """
        if detections.tracker_id is not None and self.player_manager:
            player_ids = np.array([
                self.player_manager.track_to_player.get(int(tid), -1)
                for tid in detections.tracker_id
            ])
            detections.data["player_id"] = player_ids
        
        return detections
    
    def _encode_image(self, frame: np.ndarray, format: str, quality: int) -> bytes:
        """Encode numpy array to image bytes.
        
        Args:
            frame: RGB numpy array
            format: Image format ("png" or "jpeg")
            quality: Compression quality (1-100, for JPEG)
            
        Returns:
            bytes: Encoded image data
            
        Raises:
            ValueError: If format is unsupported
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame.astype('uint8'), 'RGB')
        
        # Encode to bytes
        buffer = io.BytesIO()
        format_lower = format.lower()
        
        if format_lower == "png":
            # Use compress_level=1 for faster encoding (default is 6)
            # Lower compression = faster encoding but larger files
            pil_image.save(buffer, format="PNG", compress_level=1)
        elif format_lower == "jpeg" or format_lower == "jpg":
            pil_image.save(buffer, format="JPEG", quality=quality)
        else:
            raise ValueError(f"Unsupported image format: {format}. Use 'png' or 'jpeg'")
        
        buffer.seek(0)
        return buffer.getvalue()
    
    # ========== Public API Methods ==========
    
    def get_frame_metadata(self, frame_id: int) -> dict:
        """Get frame navigation metadata (without detection data).
        
        This method returns ONLY navigation and session metadata.
        For detection/annotation data, use get_frame_annotation_data() instead.
        This separation ensures a single source of truth for annotation data.
        
        Args:
            frame_id: Frame index to get metadata for
            
        Returns:
            dict: Frame navigation metadata (no detection data)
            
        Raises:
            ValueError: If frame_id is invalid or video not loaded
        """
        if not self.video_id:
            raise ValueError("Video ID is not set for this session")
        
        if frame_id < 0 or frame_id >= self.total_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.total_frames - 1}")
        
        return {
            "frame_id": frame_id,
            "video_id": self.video_id,
            "session_id": self.session_id,
            "has_next_frame": (frame_id + self.frame_skip_interval) < self.total_frames,
            "has_previous_frame": frame_id > 0,
            "total_frames": self.total_frames
        }
    
    def get_raw_frame_image(self, frame_id: int, format: str = "jpeg", quality: int = 85) -> bytes:
        """Get raw frame image without annotations for client-side rendering.
        
        Uses intelligent rolling cache with prefetching for optimal performance:
        - Checks cache first (instant if hit)
        - On miss: encodes synchronously
        - Prefetches next 5 frames in background
        - Keeps previous 5 frames in memory
        
        Args:
            frame_id: Frame index to retrieve
            format: Image format ("jpeg" or "png"), default "jpeg"
            quality: JPEG compression quality (1-100), default 85
            
        Returns:
            bytes: Encoded image data ready for streaming
            
        Raises:
            ValueError: If frame cannot be loaded or format invalid
        """
        if not self.video_id:
            raise ValueError("Video ID is not set for this session")
        
        if frame_id < 0 or frame_id >= self.total_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.total_frames - 1}")
        
        # Use cache with automatic prefetching
        # The cache will handle encoding if needed and trigger background prefetch
        return self.frame_cache.get(
            frame_id=frame_id,
            encode_func=self._encode_frame_for_cache,
            total_frames=self.total_frames,
            trigger_prefetch=True
        )
    
    def _encode_frame_for_cache(self, frame_id: int, format: str, quality: int) -> bytes:
        """Encode a frame for caching (used by cache prefetch).
        Ensures thread safety for decoder access by acquiring the session lock.
        """
        with self.lock:
            frame_rgb = self._load_raw_frame(frame_id)
            return self._encode_image(frame_rgb, format, quality)

    def get_frame_annotation_data(self, frame_id: int) -> Dict:
        """Get frame annotation data with Detections + RenderingConfig.
        
        Returns detection data in JSON format suitable for client rendering.
        This maintains supervision.Detections as the single source of truth.
        
        Args:
            frame_id: Frame index to get annotations for
            
        Returns:
            Dict with annotations and rendering config
            
        Raises:
            ValueError: If frame_id is invalid or video not loaded
        """
        if not self.video_id:
            raise ValueError("Video ID is not set for this session")
        
        if frame_id < 0 or frame_id >= self.total_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.total_frames - 1}")
        
        # Get detections for this frame
        detections = self._get_frame_detections(frame_id)
        
        # Initialize player manager if needed
        self._ensure_player_manager(frame_id, detections)
        
        # Apply player mapping
        detections = self._apply_player_mapping(detections)
        
        # Create default rendering config
        rendering_config = RenderingConfig.create_default()
        
        # Serialize to JSON format
        session_id_str = self.session_id if self.session_id else ""
        video_id_str = self.video_id if self.video_id else ""
        
        return create_frame_response(
            frame_id=frame_id,
            video_id=video_id_str,
            session_id=session_id_str,
            detections=detections,
            rendering_config=rendering_config,
            has_next=(frame_id + self.frame_skip_interval) < self.total_frames,
            has_previous=frame_id > 0,
            total_frames=self.total_frames
        )
    
    def get_frame_detections_with_config(self, frame_id: int) -> Tuple[Detections, RenderingConfig]:
        """Get Detections and RenderingConfig for a frame.
        
        This returns the raw Detections object and rendering config,
        useful for server-side rendering.
        
        Args:
            frame_id: Frame index
            
        Returns:
            Tuple of (Detections, RenderingConfig)
            
        Raises:
            ValueError: If frame_id is invalid or video not loaded
        """
        if not self.video_id:
            raise ValueError("Video ID is not set for this session")
        
        if frame_id < 0 or frame_id >= self.total_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.total_frames - 1}")
        
        # Get detections
        detections = self._get_frame_detections(frame_id)
        self._ensure_player_manager(frame_id, detections)
        detections = self._apply_player_mapping(detections)
        
        # Create rendering config
        rendering_config = RenderingConfig.create_default()
        
        return detections, rendering_config
    
    def update_frame_annotation_data(self, frame_id: int, annotation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update detections and rendering config for a specific frame.
        
        This method allows the web application to modify detection data (e.g., player IDs)
        and have those changes reflected in the VideoManager's in-memory detections.
        
        Args:
            frame_id: Frame index to update
            annotation_data: Dictionary with "detections" and "rendering_config" keys
            
        Returns:
            Updated annotation data as dictionary (same format as get_frame_annotation_data)
            
        Raises:
            ValueError: If frame_id is invalid, video not loaded, or data format is invalid
        """
        
        if not self.video_id:
            raise ValueError("Video ID is not set for this session")
        
        if frame_id < 0 or frame_id >= self.total_frames:
            raise ValueError(f"Invalid frame_id: {frame_id}. Must be between 0 and {self.total_frames - 1}")
        
        # Parse the incoming annotation data
        if "detections" not in annotation_data:
            raise ValueError("annotation_data must contain 'detections' key")
        
        # Convert JSON back to Detections object
        detections, rendering_config = json_to_detections(annotation_data, return_rendering_config=True)
        
        if not isinstance(detections, Detections):
            raise ValueError("Failed to parse detections from annotation_data")
        
        # Update the in-memory detections for this frame
        # We need to find which detections in self.detections belong to this frame
        # and replace them with the updated detections
        
        if self.detections is None or len(self.detections) == 0:
            # If no detections exist, just set the new ones
            self.detections = detections
        else:
            # Get the frame_id data from existing detections
            if "frame_id" in self.detections.data:
                frame_ids = self.detections.data["frame_id"]
                # Find indices for this frame
                frame_mask = frame_ids == frame_id
                
                # Get detections for other frames
                other_frames_mask = ~frame_mask
                other_detections = self.detections[other_frames_mask]
                
                # Add frame_id to new detections if not present
                if "frame_id" not in detections.data:
                    detections.data["frame_id"] = np.full(len(detections), frame_id, dtype=int)
                
                # Merge: other frames + updated frame
                if isinstance(other_detections, Detections) and len(other_detections) > 0:
                    self.detections = Detections.merge([other_detections, detections])
                else:
                    self.detections = detections
            else:
                # No frame_id tracking, replace all detections (shouldn't happen in normal use)
                self.detections = detections
        
        # Return the updated annotation data in the same format
        return self.get_frame_annotation_data(frame_id)


    def _setup_video_capture(self, video_path: str) -> dict:
        """Set up video capture and retrieve video properties.
        
        Args:
            video_path: GCS path to the video file
            
        Returns:
            dict: Video properties containing total_frames and cap object
            
        Raises:
            ValueError: If video setup fails
        """
        # GCSVideoCapture is a context manager - we need to enter it to initialize
        cap = self.storage.get_video_capture(video_path)
        cap.__enter__()  # Manually enter the context to initialize cv2.VideoCapture
        
        total_frames = cap.get("CAP_PROP_FRAME_COUNT") or 0
        
        # For some video formats, frame count is only available after reading frames
        if total_frames <= 0:
            # Try reading the first frame to initialize video properties
            ret, frame = cap.read()
            if ret:
                # Reset to beginning
                cap.set("CAP_PROP_POS_FRAMES", 0)
                # Try getting frame count again
                total_frames = cap.get("CAP_PROP_FRAME_COUNT") or 0
            else:
                pass
        
        if total_frames <= 0:
            # Clean up if we failed
            cap.__exit__(None, None, None)
            raise ValueError("Could not retrieve total frame count.")
            
        return {"cap": cap, "total_frames": total_frames}


    def _generate_session_id(self) -> str:
        """Generate a short, unique session ID.
        
        Creates a session identifier combining timestamp and random components
        for uniqueness within the 1024 concurrent session limit.
        
        Format: TTTTTTTT-RR
        - TTTTTTTT: 8-digit timestamp (seconds since epoch, mod 100M)
        - RR: 2-digit random hex (8 bits, supports 256 values)
        
        Returns:
            str: Unique session ID string (~11 characters)
        """
        with self.lock:
            timestamp = int(time.time()) % 100_000_000  # 8 digits
            random_part = random.randint(0, 0xFF)  # 8-bit random
            session_id = f"{timestamp:08d}-{random_part:02X}"
            return session_id


    def _load_players_if_json_exists(self) -> None:
        """Retrieve the PlayerManager for the video if players JSON exists."""

        if not self.video_id:
            raise ValueError("Video ID is not set, cannot load players.")

        players_path = self.path_manager.get_path("players_data_path", video_id=self.video_id)
        if not players_path:
            raise ValueError(f"Can't find players data path for video_id: {self.video_id}")

        players_json_str = self.storage.download_as_string(players_path)
        if not players_json_str:
            logger.info(f"No players data found for video: {self.video_id}")
            return

        try:
            self.player_manager = load_player_manager(self.video_id, players_json_str)
        except Exception as e:
            logger.error(f"Failed to load player manager for video: {self.video_id}: {e}")
            return


    def _load_detections(self) -> bool:
        """Load detections from Google Cloud Storage.

        Args:
            video_path: Path to the video file (used to derive detection file path)

        Returns:
            bool: True if detections were loaded successfully, False otherwise.
        """
        try:
            if not self.video_id:
                logger.warning("Video ID is not set, cannot load detections.")
                return False

            # Get the path to the detections file
            detections_path = self.path_manager.get_path("detections_path", video_id=self.video_id)
            if not detections_path:
                raise ValueError(f"No detections path found for video_id: {self.video_id}")

            # Download detections JSON from GCS
            detections_json_str = self.storage.download_as_string(detections_path)
            if not detections_json_str:
                logger.warning(f"No detections data found at path: {detections_path}")
                return False

            # Parse the JSON string
            try:
                detections_json = json.loads(detections_json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in detections file for video_id: {self.video_id}: {e}")

            # Validate that we have a list of detection data or a single detection dict
            if isinstance(detections_json, dict):
                # Single frame detection - wrap in list
                detections_json = [detections_json]
            elif not isinstance(detections_json, list):
                logger.error(f"Invalid detections format for {detections_path}: expected list or dict, got {type(detections_json)}")
                return False

            if not detections_json:  # Empty list
                logger.info(f"Empty detections list for video_id: {self.video_id}")
                self.detections = Detections.empty()
                return True

            # Convert JSON to Detections object
            detections_result = json_to_detections(detections_json, return_rendering_config=False)
            # Cast to Detections since return_rendering_config=False guarantees Detections type
            self.detections = cast(Detections, detections_result)

            # Validate the resulting detections object
            if self.detections is None or not isinstance(self.detections, Detections):
                logger.error(f"Failed to create Detections object from JSON for video_id: {self.video_id}")
                return False
            
            # Backup old tracker IDs if not already present
            if self.detections.tracker_id is not None:
                self.detections.data["old_tracker_id"] = self.detections.tracker_id

            # Check if detections have frame_index data
            if "frame_index" not in self.detections.data or len(self.detections.data["frame_index"]) == 0:
                logger.warning(f"Detections object has no valid frame_index data for video_id: {self.video_id}")
                self.detections = Detections.empty()
                return True

            logger.info(f"Successfully loaded {len(self.detections)} detections for video_id: {self.video_id}")
            return True

        except FileNotFoundError:
            logger.warning(f"Detections file not found for video_id: {self.video_id} at path: {detections_path if 'detections_path' in locals() else 'unknown'}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in detections file for video_id: {self.video_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading detections for video_id: {self.video_id}: {e}")
            return False
