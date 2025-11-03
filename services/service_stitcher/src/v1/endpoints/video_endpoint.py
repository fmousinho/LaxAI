from fastapi import HTTPException, APIRouter, Query
from fastapi.responses import StreamingResponse
from ...video_manager import VideoManager
from ..schemas.video_schema import (
    GetPlayersResponse,
    PlayerCreate,
    PlayerListItem,
    VideoLoadRequest, 
    VideoLoadResponse, 
    FrameMetadataResponse,
    AnnotationDataResponse,
    ImageFormat
)
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Tuple
import io

# Store managers with creation timestamps: session_id -> (manager, created_at)

video_managers: Dict[str, Tuple[VideoManager, float]] = {}

# Cleanup interval in seconds (12 hours)
CLEANUP_INTERVAL = 12 * 60 * 60  # 12 hours in seconds

def _log_sessions(context: str) -> None:
    try:
        logger.info(
            f"[{context}] Active sessions count={len(video_managers)} keys={list(video_managers.keys())}"
        )
    except Exception:
        # Avoid logging-related crashes
        pass

router = APIRouter(prefix="/stitcher/video", tags=["stitcher-video"])

# Log when the endpoint module loads
logger.warning("video_endpoint module loaded, router created with prefix /stitcher/video")

@router.post(
    "/load",
    response_model=VideoLoadResponse,
    summary="Loads video for stitching analysis",
    description="Start a track stitching verification session for a specific video."
)
def load (req: VideoLoadRequest) -> VideoLoadResponse:
    """Load a video from a given path or URL."""
    logger.warning(f"LOAD ENDPOINT CALLED: tenant_id={req.tenant_id}, video_path={req.video_path}")
    try:

        manager = VideoManager(req.tenant_id)
        result = manager.load_video(req.video_path)
        session_id = result.get("session_id")
        if not session_id:
            logger.error("Failed to get session ID from video manager after loading video")
            raise ValueError("Failed to get session ID from video manager")

        # Store manager with creation timestamp
        video_managers[session_id] = (manager, time.time())
        _log_sessions(f"session-registered {session_id}")

        return VideoLoadResponse(
            session_id=result["session_id"],
            total_frames=result["total_frames"],
            has_next_frame=result["has_next_frame"],
            has_previous_frame=result["has_previous_frame"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
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
    summary="Get frame navigation metadata",
    description="Get navigation and session metadata for a frame (no detection data).",
)
def get_frame_metadata(session_id: str, frame_id: int) -> FrameMetadataResponse:
    """Get navigation metadata for a specific frame (no detection/annotation data)."""
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
    img_format: ImageFormat = Query(ImageFormat.JPEG, description="Image format (png or jpeg)"),
    quality: int = Query(85, ge=1, le=100, description="JPEG quality (1-100, only for JPEG format)")
) -> StreamingResponse:
    """Get raw frame image for client-side annotation."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        manager, _ = session_data
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
        
        manager, _ = session_data
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
        
        manager, _ = session_data
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
        
        manager, _ = session_data
        # Update the detections in the video manager
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

@router.get(
    "/players/{session_id}",
    response_model=GetPlayersResponse,
    summary="Get list of players in the session",
    description="Retrieve the list of players currently tracked in the video session.",
)
def get_players(session_id: str) -> GetPlayersResponse:
    """Get the list of players for a specific session."""
    session_data = video_managers.get(session_id)
    if not session_data:
        _log_sessions(f"session-lookup-miss {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    manager, _ = session_data
    player_list = []
    players = manager.get_players()
    if not players:
        return GetPlayersResponse(players=player_list)
    for player in players:
        player = player.to_dict()
        player_list.append(PlayerListItem(**player))
    return GetPlayersResponse(players=player_list)


@router.post(
    "/player/{session_id}",
    response_model=PlayerListItem,
    summary="Add a player to the session",
    description="Add a new player to the video session."
)
def add_player(session_id: str, player_data: PlayerCreate) -> PlayerListItem:
    """Add a new player to the session."""
    session_data = video_managers.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    manager, _ = session_data
    player = manager.add_player(
        player_data.player_name if player_data.player_name else "Unknown",
        player_data.tracker_ids,
        player_number=player_data.player_number,
        image_path=player_data.image_path,
    )
    return PlayerListItem(**player.to_dict())


@router.get(
    "/player/{session_id}/{player_id}",
    response_model=PlayerListItem,
    summary="Get a player's information in the session",
    description="Retrieve a specific player's information in the video session."
)
def get_player(session_id: str, player_id: int) -> PlayerListItem:
    """Get a specific player's information in the session."""
    session_data = video_managers.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    manager, _ = session_data
    player = manager.get_player(player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return PlayerListItem(**player.to_dict())


@router.patch(
    "/player/{session_id}/{player_id}",
    response_model=PlayerListItem,
    summary="Update a player's information in the session",
    description="Update an existing player's information in the video session." 
)
def update_player(session_id: str, player_id: int, player_data: PlayerCreate) -> PlayerListItem:
    """Update an existing player's information in the session."""
    session_data = video_managers.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    manager, _ = session_data
    player = manager.update_player(
        player_id,
        name=player_data.player_name,
        tracker_ids=player_data.tracker_ids,
        image_path=player_data.image_path,
        player_number=player_data.player_number,
    )
    if not player:
        # Update failed (invalid attributes or operation). Return 400 instead of constructing an empty PlayerListItem
        # to avoid Pydantic validation errors on required fields.
        raise HTTPException(status_code=400, detail="Failed to update player")
    return PlayerListItem(**player.to_dict())

@router.delete(
    "/player/{session_id}/{player_id}",
    summary="Delete a player from the session",
    description="Delete a player from the video session."
)
def delete_player(session_id: str, player_id: int):
    """Delete a player from the session."""
    try:
        session_data = video_managers.get(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        manager, _ = session_data
        success = manager.delete_player(player_id)
        if not success:
            raise HTTPException(status_code=404, detail="Player not found")
        else:
            return
    except HTTPException:
        # Re-raise HTTPExceptions as-is (don't wrap them in 500 errors)
        raise
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