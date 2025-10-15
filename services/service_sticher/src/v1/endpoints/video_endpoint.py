from fastapi import HTTPException, APIRouter
from ...video_manager import VideoManager
from ..schemas.video_schema import VideoLoadRequest, VideoLoadResponse, VideoFrameResponse

router = APIRouter(prefix="/video", tags=["video"])

video_managers = {}

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
        video_managers[session_id] = manager    
        
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
        manager = video_managers.get(session_id)
        if not manager:
            raise HTTPException(status_code=404, detail="Session not found")

        result = manager.next_frame()
        return VideoFrameResponse(
            frame_id=result["frame_id"],
            frame_data=result["frame_data"],
            has_next_frame=result["has_next_frame"],
            has_previous_frame=result["has_previous_frame"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def previous_frame (session_id: str):
    """Go back to the previous frame in the video."""
    # Placeholder implementation
    response = {
        "success": True,
        "frame_id": 0,
        "frame_data": "base64_encoded_frame_data",
        "has_next_frame": True,
        "has_previous_frame": False,
    }
    return response

def stop_and_save (session_id: str, save_path: str):
    """Stop the session and save the processed video."""
    # Placeholder implementation
    response = {
        "success": True
    }
    return response