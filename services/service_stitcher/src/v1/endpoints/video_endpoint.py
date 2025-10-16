from fastapi import HTTPException, APIRouter
from ...video_manager import VideoManager
from ..schemas.video_schema import VideoLoadRequest, VideoLoadResponse, VideoFrameResponse
import asyncio
import time
from typing import Dict, Tuple
from contextlib import asynccontextmanager

# Store managers with creation timestamps: session_id -> (manager, created_at)
video_managers: Dict[str, Tuple[VideoManager, float]] = {}

# Cleanup interval in seconds (12 hours)
CLEANUP_INTERVAL = 12 * 60 * 60  # 12 hours in seconds

async def cleanup_expired_sessions():
    """Background task to clean up expired video sessions."""
    while True:
        try:
            current_time = time.time()
            expired_sessions = []

            # Find expired sessions (older than 12 hours)
            for session_id, (manager, created_at) in video_managers.items():
                if current_time - created_at > CLEANUP_INTERVAL:
                    expired_sessions.append(session_id)

            # Remove expired sessions
            for session_id in expired_sessions:
                del video_managers[session_id]

            # Log cleanup if any sessions were removed
            if expired_sessions:
                print(f"Cleaned up {len(expired_sessions)} expired video sessions")

        except Exception as e:
            print(f"Error during session cleanup: {e}")

        # Wait 1 hour before next cleanup check
        await asyncio.sleep(60 * 60)  # 1 hour

@asynccontextmanager
async def lifespan(app):
    """Handle application startup and shutdown events."""
    # Startup: start the cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    
    yield
    
    # Shutdown: cancel the cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

router = APIRouter(prefix="/video", tags=["video"], lifespan=lifespan)

@router.post(
    "/load",
    response_model=VideoLoadResponse,
    summary="Loads video for stitching analysis",
    description="Start a track stitching verification session for a specific video."
)
def load (req: VideoLoadRequest) -> VideoLoadResponse:
    """Load a video from a given path or URL."""
    try:
        manager = VideoManager(req.tenant_id)
        result = manager.load_video(req.video_path)
        session_id = result.get("session_id")
        if not session_id:
            raise ValueError("Failed to get session ID from video manager")
        
        # Store manager with creation timestamp
        video_managers[session_id] = (manager, time.time())
        
        return VideoLoadResponse(
            session_id=result["session_id"],
            video_path=result["video_path"],
            total_frames=result["total_frames"],
            has_next_frame=result["has_next_frame"],
            has_previous_frame=result["has_previous_frame"],
        )
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/next-frame/{session_id}",
    response_model=VideoFrameResponse,
    summary="Advances to the next frame after skips",
    description="Advance to the next frame in the loaded video session.",
)
def next_frame(session_id: str) -> VideoFrameResponse:
    """Advance to the next frame in the video."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _ = session_data
        result = manager.next_frame()
        return VideoFrameResponse(
            frame_id=result["frame_id"],
            frame_data=result["frame_data"],
            has_next_frame=result["has_next_frame"],
            has_previous_frame=result["has_previous_frame"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/previous-frame/{session_id}",
    response_model=VideoFrameResponse,
    summary="Rewinds to the previous frame",
    description="Rewind to the previous frame in the loaded video session.",
)
def previous_frame (session_id: str) -> VideoFrameResponse:
    """Go back to the previous frame in the video."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        manager, _ = session_data
        result = manager.previous_frame()
        return VideoFrameResponse(
            frame_id=result["frame_id"],
            frame_data=result["frame_data"],
            has_next_frame=result["has_next_frame"],
            has_previous_frame=result["has_previous_frame"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete(
    "/stop-and-save/{session_id}",
    summary="Stops the session and saves the processed video",
    description="Stop the session and save the processed video.",
)
def stop_and_save (session_id: str):
    """Stop the session and save the processed video."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        del video_managers[session_id]
    except HTTPException:
        # Re-raise HTTPExceptions as-is (don't wrap them in 500 errors)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return