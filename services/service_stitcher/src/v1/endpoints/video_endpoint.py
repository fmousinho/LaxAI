from fastapi import HTTPException, APIRouter, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from ...video_manager import VideoManager
from ..schemas.video_schema import (
    VideoLoadRequest, 
    VideoLoadResponse, 
    FrameMetadataResponse,
    AnnotationRecipeResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
    ImageFormat
)
import asyncio
import time
import uuid
from typing import Dict, Tuple
from contextlib import asynccontextmanager
import io

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
        
        manager, _ = session_data
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

        manager, _ = session_data
        result = manager.previous_frame()
        return FrameMetadataResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/frames/{session_id}/{frame_id}/metadata",
    response_model=FrameMetadataResponse,
    summary="Get frame metadata",
    description="Get metadata for a specific frame including detections and player mappings.",
)
def get_frame_metadata(session_id: str, frame_id: int) -> FrameMetadataResponse:
    """Get metadata for a specific frame."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _ = session_data
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
    format: ImageFormat = Query(ImageFormat.PNG, description="Image format (png or jpeg)"),
    quality: int = Query(95, ge=1, le=100, description="JPEG quality (1-100, only for JPEG format)")
) -> StreamingResponse:
    """Get raw frame image for client-side annotation."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _ = session_data
        
        # Get encoded image bytes
        image_bytes = manager.get_raw_frame_image(frame_id, format.value, quality)
        
        # Determine media type
        media_type = "image/png" if format == ImageFormat.PNG else "image/jpeg"
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Content-Disposition": f"inline; filename=frame_{frame_id}.{format.value}"
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/frames/{session_id}/{frame_id}/recipe",
    response_model=AnnotationRecipeResponse,
    summary="Get annotation recipe for frame",
    description="Get declarative annotation recipe that can be rendered on client or server.",
)
def get_frame_annotation_recipe(
    session_id: str,
    frame_id: int
) -> AnnotationRecipeResponse:
    """Get annotation recipe for a specific frame."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _ = session_data
        recipe = manager.get_frame_annotation_recipe(frame_id)
        
        return AnnotationRecipeResponse(
            frame_id=frame_id,
            recipe=recipe.to_dict()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{session_id}/generate-video",
    response_model=VideoGenerationResponse,
    summary="Generate annotated video with user edits",
    description="Queue video generation task using user-edited annotation recipes.",
)
async def generate_annotated_video(
    session_id: str,
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks
) -> VideoGenerationResponse:
    """Generate video with client-modified annotations."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _ = session_data
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Queue background task for video generation
        background_tasks.add_task(
            render_video_with_recipes,
            manager,
            task_id,
            request.frame_recipes,
            request.output_format,
            request.fps
        )
        
        return VideoGenerationResponse(
            task_id=task_id,
            status="queued",
            message=f"Video generation task {task_id} queued successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def render_video_with_recipes(
    manager: VideoManager,
    task_id: str,
    frame_recipes: Dict[int, Dict],
    output_format: str,
    fps: int
):
    """Background task to render final video with annotation recipes.
    
    Args:
        manager: VideoManager instance
        task_id: Unique task identifier
        frame_recipes: Dictionary mapping frame_id to recipe dict
        output_format: Output video format (mp4, avi, etc.)
        fps: Frames per second
    """
    import cv2
    from shared_libs.common.annotation_schema import AnnotationRecipe
    from services.service_stitcher.src.annotation_renderer import AnnotationRenderer
    
    try:
        renderer = AnnotationRenderer()
        output_path = f"{manager.storage.user_id}/rendered/{task_id}.{output_format}"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v' if output_format == 'mp4' else 'XVID')
        
        # Get first frame to determine dimensions
        first_frame = manager._load_raw_frame(0)
        height, width = first_frame.shape[:2]
        
        # Create temporary local file
        temp_output = f"/tmp/{task_id}.{output_format}"
        out = cv2.VideoWriter(temp_output, fourcc, float(fps), (width, height))
        
        # Process each frame with its recipe
        for frame_id in sorted(frame_recipes.keys()):
            try:
                # Load raw frame
                frame_rgb = manager._load_raw_frame(frame_id)
                
                # Parse recipe from dict
                recipe_dict = frame_recipes[frame_id]
                recipe = AnnotationRecipe.from_dict(recipe_dict)
                
                # Render annotations
                annotated_rgb = renderer.render_recipe(frame_rgb, recipe)
                
                # Convert to BGR for video writer
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(annotated_bgr)
                
            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")
                continue
        
        # Release video writer
        out.release()
        
        # Upload to GCS
        manager.storage.upload_file(temp_output, output_path)
        
        # Clean up temporary file
        import os
        os.remove(temp_output)
        
        print(f"Video generation task {task_id} completed: {output_path}")
        
    except Exception as e:
        print(f"Error in video generation task {task_id}: {e}")


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