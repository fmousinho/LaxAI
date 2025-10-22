from fastapi import HTTPException, APIRouter, Query
from fastapi.responses import StreamingResponse
from ...video_manager import VideoManager
from ..schemas.video_schema import (
    VideoLoadRequest, 
    VideoLoadResponse, 
    FrameMetadataResponse,
    AnnotationDataResponse,
    ImageFormat
)
import logging
import asyncio
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Tuple
from contextlib import asynccontextmanager
import io

# Store managers with creation timestamps and a lock: session_id -> (manager, created_at, lock)
from threading import Lock

video_managers: Dict[str, Tuple[VideoManager, float, Lock]] = {}

# Cleanup interval in seconds (12 hours)
CLEANUP_INTERVAL = 12 * 60 * 60  # 12 hours in seconds

async def cleanup_expired_sessions():
    """Background task to clean up expired video sessions."""
    while True:
        try:
            current_time = time.time()
            expired_sessions = []
            # Find expired sessions (older than 12 hours)
            for session_id, (manager, created_at, lock) in list(video_managers.items()):
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

router = APIRouter(prefix="/stitcher/video", tags=["stitcher-video"], lifespan=lifespan)

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
            logger.error("Failed to get session ID from video manager after loading video")
            raise ValueError("Failed to get session ID from video manager")
        
        # Store manager with creation timestamp
        video_managers[session_id] = (manager, time.time(), Lock())
        
        return VideoLoadResponse(
            session_id=result["session_id"],
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
    response_model=FrameMetadataResponse,
    summary="Advances to the next frame after skips",
    description="Advance to the next frame in the loaded video session and return metadata.",
)
def next_frame(session_id: str) -> FrameMetadataResponse:
    """Advance to the next frame in the video and return frame metadata."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _, lock = session_data
        with lock:
            result = manager.next_frame()
        return FrameMetadataResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/previous-frame/{session_id}",
    response_model=FrameMetadataResponse,
    summary="Rewinds to the previous frame",
    description="Rewind to the previous frame in the loaded video session and return metadata.",
)
def previous_frame (session_id: str) -> FrameMetadataResponse:
    """Go back to the previous frame in the video and return frame metadata."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        manager, _, lock = session_data
        with lock:
            result = manager.previous_frame()
        return FrameMetadataResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/frames/{session_id}/{frame_id}/metadata",
    response_model=FrameMetadataResponse,
    summary="Get frame navigation metadata",
    description="Get navigation and session metadata for a frame (no detection data). Use /recipe endpoint for annotations.",
)
def get_frame_metadata(session_id: str, frame_id: int) -> FrameMetadataResponse:
    """Get navigation metadata for a specific frame (no detection/annotation data)."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _, lock = session_data
        with lock:
            result = manager.get_frame_metadata(frame_id)
        return FrameMetadataResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/frames/{session_id}/{frame_id}/image",
    summary="Get raw frame image",
    description="Stream raw frame image in PNG or JPEG format for client-side annotation.",
)
def get_frame_image(
    session_id: str, 
    frame_id: int,
    img_format: ImageFormat = Query(ImageFormat.JPEG, description="Image format (png or jpeg)"),
    quality: int = Query(85, ge=1, le=100, description="JPEG quality (1-100, only for JPEG format)")
) -> StreamingResponse:
    """Get raw frame image for client-side annotation."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _, lock = session_data
        with lock:
            image_bytes = manager.get_raw_frame_image(frame_id, img_format.value, quality)
        media_type = "image/png" if img_format == ImageFormat.PNG else "image/jpeg"
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Content-Disposition": f"inline; filename=frame_{frame_id}.{img_format.value}"
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_frame_image for session {session_id}, frame {frame_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/frames/{session_id}/cache-stats",
    summary="Get frame cache statistics",
    description="Get performance statistics for the rolling frame cache (hit rate, prefetch effectiveness, etc.)",
)
def get_cache_stats(session_id: str) -> dict:
    """Get cache performance statistics for a session."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _, lock = session_data
        with lock:
            stats = manager.frame_cache.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/frames/{session_id}/{frame_id}/annotations",
    response_model=AnnotationDataResponse,
    summary="Get frame annotation data (detections + rendering config)",
    description="Get supervision.Detections and rendering configuration for a frame. This is the single source of truth for annotation data.",
)
def get_frame_annotations(
    session_id: str,
    frame_id: int
) -> AnnotationDataResponse:
    """Get annotation data for a specific frame (detections + rendering config)."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            logger.error(f"Session ID {session_id} not found when fetching annotations for frame {frame_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _, lock = session_data
        with lock:
            annotation_data = manager.get_frame_annotation_data(frame_id)
        return AnnotationDataResponse(**annotation_data)
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in get_frame_annotations for session {session_id}, frame {frame_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/frames/{session_id}/{frame_id}/annotations",
    response_model=AnnotationDataResponse,
    summary="Update frame annotation data",
    description="Update detections and rendering configuration for a frame. This allows the web app to modify player mappings and other annotation data.",
)
def update_frame_annotations(
    session_id: str,
    frame_id: int,
    annotation_data: AnnotationDataResponse
) -> AnnotationDataResponse:
    """Update annotation data for a specific frame.
    
    The web application can use this to update player IDs, modify detections,
    or change rendering styles. The updated data is stored in the VideoManager's
    in-memory detections.
    """
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _, lock = session_data
        # Update the detections in the video manager
        with lock:
            updated_data = manager.update_frame_annotation_data(
                frame_id=frame_id,
                annotation_data=annotation_data.model_dump()
            )
        return AnnotationDataResponse(**updated_data)
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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