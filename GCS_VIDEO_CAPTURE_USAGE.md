# GCSVideoCapture Usage Guide

The `GCSVideoCapture` class provides an in-memory approach to working with video files stored in Google Cloud Storage. It downloads the video data into memory and provides a cv2-compatible interface.

## Key Features

- **In-memory processing**: Downloads video data as bytes into memory
- **Context manager**: Automatically handles resource cleanup
- **cv2 compatibility**: Provides all standard cv2.VideoCapture methods
- **Temporary file management**: Uses efficient temporary file handling for cv2 compatibility

## Usage Example

```python
from common.google_storage import get_storage
import cv2

# Initialize storage client
storage_client = get_storage("tenant1")

# Use the video capture context manager
video_blob_name = "raw/sample_video.mp4"

with storage_client.get_video_capture(video_blob_name) as video_cap:
    if video_cap.isOpened():
        # Get video properties
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        # Process frames
        frame_number = 0
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
                
            # Process the frame (frame is a numpy array)
            print(f"Processing frame {frame_number}, shape: {frame.shape}")
            
            # You can also seek to specific frames
            if frame_number == 10:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 50)  # Jump to frame 50
            
            frame_number += 1
            
            # Break after processing a few frames for demo
            if frame_number > 5:
                break
                
    else:
        print("Failed to open video")
```

## Available Methods

- `read()`: Read the next frame, returns (ret, frame)
- `get(prop_id)`: Get video property (e.g., cv2.CAP_PROP_FRAME_COUNT)
- `set(prop_id, value)`: Set video property (e.g., cv2.CAP_PROP_POS_FRAMES)
- `isOpened()`: Check if video is successfully opened
- `release()`: Manually release resources (handled automatically by context manager)

## Resource Management

The class automatically:
1. Downloads video data into memory as bytes
2. Creates a temporary file for cv2 compatibility
3. Cleans up temporary files and releases resources when exiting the context
4. Handles errors gracefully with proper cleanup

## Memory Considerations

- Video data is loaded entirely into memory first
- A temporary file is still created for cv2 compatibility (cv2 requires file paths)
- The temporary file is automatically deleted when the context exits
- For very large videos, consider the available system memory
