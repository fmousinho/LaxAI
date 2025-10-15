import uuid
import time
import random
import threading
import typing

import shared_libs.config.logging_config 
from shared_libs.common.google_storage import get_storage

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
        current_frame (int): Current frame position in the video
        total_frames (int): Total number of frames in the loaded video
        frame_skip_interval (int): Number of frames to skip between reads for performance
    """

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
        self.current_frame = None
        self.total_frames = 0
        self.frame_skip_interval = frame_skip_interval

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
            # GCSVideoCapture is a context manager - we need to enter it to initialize
            self.cap = self.storage.get_video_capture(video_path)
            self.cap.__enter__()  # Manually enter the context to initialize cv2.VideoCapture
            
            self.total_frames = self.cap.get("CAP_PROP_FRAME_COUNT") or 0
            
            # For some video formats, frame count is only available after reading frames
            if self.total_frames <= 0:
                # Try reading the first frame to initialize video properties
                ret, frame = self.cap.read()
                if ret:
                    # Reset to beginning
                    self.cap.set("CAP_PROP_POS_FRAMES", 0)
                    # Try getting frame count again
                    self.total_frames = self.cap.get("CAP_PROP_FRAME_COUNT") or 0
                else:
                    pass
            
            if self.total_frames <= 0:
                # Clean up if we failed
                self.cap.__exit__(None, None, None)
                self.cap = None
                raise ValueError("Could not retrieve total frame count.")
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
        if self.current_frame is None:
            self.current_frame = 0
        else:
            self.current_frame += self.frame_skip_interval
            if self.current_frame >= self.total_frames:
                raise ValueError("End of video reached")
            if not self.cap.set("CAP_PROP_POS_FRAMES", self.current_frame):
                raise ValueError(f"Failed to set frame position to {self.current_frame}")

        # Read the frame
        success, frame_data = self.cap.read(return_format="rgb")
        if not success or frame_data is None:
            raise ValueError(f"Failed to read frame at position {self.current_frame}")
        
        # Check if there are more frames available
        has_next_frame = (self.current_frame + self.frame_skip_interval) < self.total_frames
        
        result = {
            "frame_id": self.current_frame,
            "frame_data": frame_data,
            "has_next_frame": has_next_frame,
            "has_previous_frame": self.current_frame > 0,
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
        if self.current_frame is None:
            self.current_frame = 0
        else:
            self.current_frame -= self.frame_skip_interval
            if self.current_frame < 0:
                self.current_frame = 0
            if not self.cap.set("CAP_PROP_POS_FRAMES", self.current_frame):
                raise ValueError(f"Failed to set frame position to {self.current_frame}")

        # Read the frame
        success, frame_data = self.cap.read(return_format="rgb")
        if not success or frame_data is None:
            raise ValueError(f"Failed to read frame at position {self.current_frame}")
        
        result = {
            "frame_id": self.current_frame,
            "frame_data": frame_data,
            "has_next_frame": (self.current_frame + self.frame_skip_interval) < self.total_frames,
            "has_previous_frame": self.current_frame > 0,
        }

        return result


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