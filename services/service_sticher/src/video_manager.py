import os
import json
import time
import random
import threading
import typing
import logging
import concurrent.futures
import numpy as np
from supervision import Detections

import shared_libs.config.logging_config 
from shared_libs.common.player_manager import initialize_player_manager, load_player_manager
from shared_libs.common.google_storage import get_storage, GCSPaths
from shared_libs.common.detection_utils import json_to_detections, detections_to_json

try:
    from shared_libs.common.detection import DetectionModel
except ImportError:
    raise ImportError("DetectionModel could not be imported. Ensure the module is available.")

try:
    from shared_libs.common.tracker import AffineAwareByteTrack
except ImportError:
    raise ImportError("AffineAwareByteTrack could not be imported. Ensure the module is available.")


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
    detector = DetectionModel()
    tracker = AffineAwareByteTrack()

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

    def __del__(self):
        """Cleanup method to properly close video capture when VideoManager is destroyed."""
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
            self.video_id = os.path.basename(video_path).split(".")[0]

            # Run detections loading and video capture setup in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                detections_future = executor.submit(self._load_detections, self.video_id)
                capture_future = executor.submit(self._setup_video_capture, video_path)
                players_future = executor.submit(self._load_players_if_json_exists, self.video_id)
                
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
                "video_path": video_path,
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
        """Advance to the next frame in the video using frame skipping.
        
        Reads the next frame from the video capture object, advancing by
        frame_skip_interval frames (default 30) for performance optimization.
        
        Returns:
            dict: Frame advancement result containing:
                - frame_id (int): The current frame index after advancement
                - frame_data: numpy array frame image data (RGB format)
                - has_next_frame (bool): Whether there are more frames to read
                - has_previous_frame (bool): Whether there are pr   evious frames
                
        Raises:
            ValueError: If no video is loaded or frame reading fails
        """
        if not self.cap:
            raise ValueError("No video loaded for this session.")
        
        # Finds the next frame position and sets the cap
        if self.current_frame_id is None:
            self.current_frame_id = 0
            self.player_manager = initialize_player_manager(self.video_id, )
        else:
            self.current_frame_id += self.frame_skip_interval
            if self.current_frame_id >= self.total_frames:
                raise ValueError("End of video reached")
            if not self.cap.set("CAP_PROP_POS_FRAMES", self.current_frame_id):
                raise ValueError(f"Failed to set frame position to {self.current_frame_id}")

        # Read the frame
        success, frame_data = self.cap.read(return_format="rgb")
        if not success or frame_data is None:
            raise ValueError(f"Failed to read frame at position {self.current_frame_id}")
        mask = self.detections.data["frame_index"] == self.current_frame_id
        frame_detections = self.detections[mask]

        if not self.player_manager:
            self.player_manager = initialize_player_manager(self.video_id, self.current_frame_id, frame_detections)
    

        # Check if there are more frames available
        has_next_frame = (self.current_frame_id + self.frame_skip_interval) < self.total_frames

        result = {
            "frame_id": self.current_frame_id,
            "frame_data": frame_data,
            "has_next_frame": has_next_frame,
            "has_previous_frame": self.current_frame_id > 0,
        }

        return result



    def previous_frame(self) -> dict:        
        """Go back to the previous frame in the video.
        
        Moves the current frame position back by frame_skip_interval frames
        (default 30) and reads that frame from the video capture object.
        
        Returns:
            dict: Frame rewind result containing:
                - frame_id (int): The current frame index after rewinding
                - frame_data: numpy array frame image data (RGB format)
        """
        if not self.cap:
            raise ValueError("No video loaded for this session.")
        
        # Finds the previous frame position and sets the cap
        if self.current_frame_id is None:
            self.current_frame_id = 0
        else:
            self.current_frame_id -= self.frame_skip_interval
            if self.current_frame_id < 0:
                self.current_frame_id = 0
            if not self.cap.set("CAP_PROP_POS_FRAMES", self.current_frame_id):
                raise ValueError(f"Failed to set frame position to {self.current_frame_id}")

        # Read the frame
        success, frame_data = self.cap.read(return_format="rgb")
        if not success or frame_data is None:
            raise ValueError(f"Failed to read frame at position {self.current_frame_id}")
        
        result = {
            "frame_id": self.current_frame_id,
            "frame_data": frame_data,
            "has_next_frame": (self.current_frame_id + self.frame_skip_interval) < self.total_frames,
            "has_previous_frame": self.current_frame_id > 0,
        }

        return result


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


    def _load_players_if_json_exists(self, video_id: str) -> None:
        """Retrieve the PlayerManager for the video if players JSON exists."""
        players_path = self.path_manager.get_path("players_data_path", video_id=video_id)
        if not players_path:
            raise ValueError(f"Can't find players data path for video_id: {video_id}")

        players_json_str = self.storage.download_as_string(players_path)
        if not players_json_str:
            logger.info(f"No players data found for video: {video_id}")
            return 

        try:
            players_json = json.loads(players_json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse players JSON, creating new manager for: {video_id}: {e}")
            return
        
        try:
            self.player_manager = load_player_manager(players_json)
        except Exception as e:
            logger.error(f"Failed to load player manager for video: {video_id}: {e}")
            return


    def _load_detections(self, video_path: str) -> bool:
        """Load detections from Google Cloud Storage.

        Args:
            video_path: Path to the video file (used to derive detection file path)

        Returns:
            bool: True if detections were loaded successfully, False otherwise.
        """
        try:

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
            self.detections = json_to_detections(detections_json)

            # Validate the resulting detections object
            if self.detections is None or not isinstance(self.detections, Detections):
                logger.error(f"Failed to create Detections object from JSON for video_id: {self.video_id}")
                return False

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
