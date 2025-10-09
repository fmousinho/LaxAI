"""
DataPrfrom ..schemas.dataprep import (
    ProcessFoldersResponse,
    StartPrepRequest,
    StartPrepResponse,
    VerificationImagesResponse,
    RecordResponseRequest,
    RecordResponseResponse,
    SaveGraphResponse,
    SaveGraphImageResponse,
    SuspendPrepResponse,
    MoveCropsResponse,
    ErrorResponse,
)ints for track stitching verification workflows.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse

from ..schemas.dataprep import (
    ProcessFoldersResponse,
    StartPrepRequest,
    StartPrepResponse,
    VerificationImagesResponse,
    RecordResponseRequest,
    RecordResponseResponse,
    SaveGraphResponse,
    SaveGraphImageResponse,
    SuspendPrepResponse,
    MoveCropsResponse,
    SplitTrackRequest,
    SplitTrackResponse,
    GraphStatisticsResponse,
    ErrorResponse,
)
from ...workflows.manager import DataPrepManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dataprep", tags=["dataprep"])

# Global manager instances keyed by tenant_id
# In production, consider using a proper session/cache management system
_managers: Dict[str, DataPrepManager] = {}


def get_manager(tenant_id: str) -> DataPrepManager:
    """Get or create a DataPrepManager for the given tenant."""
    if tenant_id not in _managers:
        _managers[tenant_id] = DataPrepManager(tenant_id)
        logger.info(f"Created DataPrepManager for tenant: {tenant_id}")
    return _managers[tenant_id]


async def save_all_active_graphs():
    """Save graphs for all active DataPrepManager instances."""
    logger.info("Attempting to save graphs for all active sessions...")
    saved_count = 0
    failed_count = 0
    
    for tenant_id, manager in _managers.items():
        try:
            if manager.stitcher is not None and manager.current_video_id is not None:
                logger.info(f"Saving graph for tenant {tenant_id}, video {manager.current_video_id}")
                if manager.save_graph():
                    saved_count += 1
                    logger.info(f"Successfully saved graph for tenant {tenant_id}")
                else:
                    failed_count += 1
                    logger.error(f"Failed to save graph for tenant {tenant_id}")
            else:
                logger.debug(f"No active session for tenant {tenant_id}, skipping save")
        except Exception as e:
            failed_count += 1
            logger.error(f"Error saving graph for tenant {tenant_id}: {e}")
    
    logger.info(f"Graph save operation complete: {saved_count} saved, {failed_count} failed")
    return saved_count, failed_count


@router.get(
    "/folders",
    response_model=ProcessFoldersResponse,
    summary="List Process Folders",
    description="Get a list of all available process folders (video IDs) for verification."
)
async def get_process_folders(tenant_id: str) -> ProcessFoldersResponse:
    """
    List all process folders available for the tenant.

    Args:
        tenant_id: The tenant identifier

    Returns:
        List of process folder names
    """
    try:
        manager = get_manager(tenant_id)
        folders = manager.get_process_folders()
        return ProcessFoldersResponse(folders=folders)
    except Exception as e:
        logger.error(f"Failed to get process folders for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list process folders: {str(e)}")


@router.post(
    "/start",
    response_model=StartPrepResponse,
    summary="Start Verification Session",
    description="Start a track stitching verification session for a specific video."
)
async def start_prep(request: StartPrepRequest, tenant_id: str) -> StartPrepResponse:
    """
    Start a verification session for the specified process folder.

    Args:
        request: Request containing the process folder name
        tenant_id: The tenant identifier

    Returns:
        Success status and optional message
    """
    try:
        manager = get_manager(tenant_id)
        success = manager.start_prep(request.video_id)

        if success:
            return StartPrepResponse(success=True, message=f"Started verification session for {request.video_id}")
        else:
            return StartPrepResponse(success=False, message=f"Failed to start session for {request.video_id}")

    except Exception as e:
        logger.error(f"Failed to start prep for tenant {tenant_id}, folder {request.video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start verification session: {str(e)}")


@router.get(
    "/verify",
    response_model=VerificationImagesResponse,
    summary="Get Verification Images",
    description="Get the next pair of track groups for manual verification."
)
async def get_verification_images(tenant_id: str) -> VerificationImagesResponse:
    """
    Get the next pair of images for verification.

    Args:
        tenant_id: The tenant identifier

    Returns:
        Verification data including group IDs and GCS prefixes
    """
    try:
        manager = get_manager(tenant_id)
        result = manager.get_images_for_verification()

        # Convert the dict result to the response model
        return VerificationImagesResponse(**result)

    except Exception as e:
        logger.error(f"Failed to get verification images for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get verification images: {str(e)}")


@router.post(
    "/respond",
    response_model=RecordResponseResponse,
    summary="Record Verification Response",
    description="Record a user's verification decision for the current pair."
)
async def record_response(request: RecordResponseRequest, tenant_id: str) -> RecordResponseResponse:
    """
    Record a verification response.

    Args:
        request: Request containing the user's decision
        tenant_id: The tenant identifier

    Returns:
        Success status and optional message
    """
    try:
        manager = get_manager(tenant_id)
        logger.info(
            f"[DATAPREP] Received record_response request: pair_id={request.pair_id!r} (type={type(request.pair_id).__name__}), "
            f"decision={request.decision}, tenant_id={tenant_id}"
        )
        result = manager.record_response(request.pair_id, request.decision)
        logger.info(f"[DATAPREP] record_response result: {result}")
        return RecordResponseResponse(**result)

    except Exception as e:
        logger.error(f"Failed to record response for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record response: {str(e)}")


@router.post(
    "/save-graph",
    response_model=SaveGraphResponse,
    summary="Save Graph State",
    description="Save the current graph state to GCS for persistence."
)
async def save_graph(tenant_id: str) -> SaveGraphResponse:
    """
    Save the current graph state to GCS.

    Args:
        tenant_id: The tenant identifier

    Returns:
        Success status and optional message
    """
    try:
        manager = get_manager(tenant_id)
        success = manager.save_graph()

        if success:
            return SaveGraphResponse(success=True, message="Graph saved successfully")
        else:
            return SaveGraphResponse(success=False, message="Failed to save graph")

    except Exception as e:
        logger.error(f"Failed to save graph for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save graph: {str(e)}")


@router.post(
    "/save-graph-image",
    response_model=SaveGraphImageResponse,
    summary="Save Graph Visualization",
    description="Generate and save a visual representation of the track relationship graph."
)
async def save_graph_image(tenant_id: str) -> SaveGraphImageResponse:
    """
    Generate and save a graph visualization image to GCS.

    Args:
        tenant_id: The tenant identifier

    Returns:
        Success status and optional message
    """
    try:
        manager = get_manager(tenant_id)
        success, image_url = manager.save_graph_image()

        if success:
            return SaveGraphImageResponse(
                success=True, 
                message="Graph image saved successfully",
                image_url=image_url
            )
        else:
            return SaveGraphImageResponse(success=False, message="Failed to save graph image", image_url=None)

    except Exception as e:
        logger.error(f"Failed to save graph image for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save graph image: {str(e)}")


@router.get(
    "/graph-statistics",
    response_model=GraphStatisticsResponse,
    summary="Get Graph Statistics",
    description="Get statistics and summary information about the current track relationship graph."
)
async def get_graph_statistics(tenant_id: str) -> GraphStatisticsResponse:
    """
    Get statistics about the current graph state.

    Args:
        tenant_id: The tenant identifier

    Returns:
        Graph statistics including track counts, relationships, and player groups
    """
    try:
        manager = get_manager(tenant_id)
        
        if manager.stitcher is None:
            return GraphStatisticsResponse(
                success=False, 
                message="No active stitching session",
                total_tracks=None,
                total_relationships=None,
                player_count=None,
                player_groups=None,
                verification_mode=None
            )
        
        # Get graph data from the stitcher
        graph_data = manager.stitcher.export_graph_data()
        
        return GraphStatisticsResponse(
            success=True,
            message="Graph statistics retrieved successfully",
            total_tracks=graph_data['metadata']['total_tracks'],
            total_relationships=graph_data['metadata']['total_relationships'],
            player_count=graph_data['metadata']['player_count'],
            player_groups=graph_data['player_groups'],
            verification_mode=graph_data['metadata']['verification_mode']
        )

    except Exception as e:
        logger.error(f"Failed to get graph statistics for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph statistics: {str(e)}")


@router.post(
    "/suspend",
    response_model=SuspendPrepResponse,
    summary="Suspend Verification Session",
    description="Save the current session state and suspend the verification process."
)
async def suspend_prep(tenant_id: str) -> SuspendPrepResponse:
    """
    Suspend the current verification session.

    Args:
        tenant_id: The tenant identifier

    Returns:
        Success status and optional message
    """
    try:
        manager = get_manager(tenant_id)
        success = manager.suspend_prep()

        if success:
            return SuspendPrepResponse(success=True, message="Session suspended successfully")
        else:
            return SuspendPrepResponse(success=False, message="Failed to suspend session")

    except Exception as e:
        logger.error(f"Failed to suspend prep for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to suspend session: {str(e)}")


@router.post(
    "/move-crops",
    response_model=MoveCropsResponse,
    summary="Move Crops to Verified Tracks",
    description="Move crops from unverified_tracks to verified_tracks based on track graph associations."
)
async def move_crops_to_verified(tenant_id: str) -> MoveCropsResponse:
    """
    Move crops from unverified to verified tracks based on graph associations.

    Args:
        tenant_id: The tenant identifier

    Returns:
        Success status and optional message
    """
    try:
        manager = get_manager(tenant_id)
        success = manager.move_crops_to_verified()

        if success:
            return MoveCropsResponse(success=True, message="Crops moved to verified tracks successfully")
        else:
            return MoveCropsResponse(success=False, message="Failed to move some or all crops")

    except Exception as e:
        logger.error(f"Failed to move crops for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to move crops: {str(e)}")


@router.post(
    "/split-track",
    response_model=SplitTrackResponse,
    summary="Split Track at Frame",
    description="Split a track into two parts at the specified frame boundary where a player switch occurred."
)
async def split_track_at_frame(request: SplitTrackRequest, tenant_id: str) -> SplitTrackResponse:
    """
    Split a track that was incorrectly merged by the tracker.

    This endpoint allows correcting cases where the tracker incorrectly grouped
    two different players in the same track. The track is split at the frame
    where the player switch occurs.

    Args:
        request: Split track request containing track_id and crop_image_name
        tenant_id: The tenant identifier

    Returns:
        Response indicating success or failure of the split operation
    """
    try:
        manager = get_manager(tenant_id)
        
        # Validate that the crop_image_name is a valid filename
        crop_image_name = request.crop_image_name
        import string
        if not crop_image_name or len(crop_image_name) > 255:
            raise HTTPException(
                status_code=400, 
                detail="Malformed request: crop_image_name must be a valid filename (1-255 characters)"
            )
        
        # Check for invalid characters (allow alphanumeric, dots, underscores, hyphens)
        valid_chars = string.ascii_letters + string.digits + '._-'
        if not all(c in valid_chars for c in crop_image_name):
            raise HTTPException(
                status_code=400,
                detail="Malformed request: crop_image_name contains invalid characters. Only letters, numbers, dots, underscores, and hyphens are allowed"
            )
        
        # Check for path separators or query parameters
        if '/' in crop_image_name or '\\' in crop_image_name or '?' in crop_image_name:
            raise HTTPException(
                status_code=400,
                detail="Malformed request: crop_image_name cannot contain path separators or query parameters"
            )
        
        success = manager.split_track_at_frame(request.track_id, crop_image_name)

        if success:
            return SplitTrackResponse(
                success=True,
                message=f"Track {request.track_id} successfully split at frame from {crop_image_name}"
            )
        else:
            return SplitTrackResponse(
                success=False,
                message=f"Failed to split track {request.track_id}"
            )

    except Exception as e:
        logger.error(f"Failed to split track for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to split track: {str(e)}")