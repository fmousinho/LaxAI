"""
DataPrep API endpoints for track stitching verification workflows.
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
    ErrorResponse
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
            if manager.stitcher is not None and manager.current_process_folder is not None:
                logger.info(f"Saving graph for tenant {tenant_id}, process folder {manager.current_process_folder}")
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
        success = manager.start_prep(request.process_folder)

        if success:
            return StartPrepResponse(success=True, message=f"Started verification session for {request.process_folder}")
        else:
            return StartPrepResponse(success=False, message=f"Failed to start session for {request.process_folder}")

    except Exception as e:
        logger.error(f"Failed to start prep for tenant {tenant_id}, folder {request.process_folder}: {e}")
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
        success = manager.record_response(request.decision)

        if success:
            return RecordResponseResponse(success=True, message=f"Recorded decision: {request.decision}")
        else:
            return RecordResponseResponse(success=False, message="Failed to record response")

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
        success = manager.save_graph_image()

        if success:
            return SaveGraphImageResponse(success=True, message="Graph image saved successfully")
        else:
            return SaveGraphImageResponse(success=False, message="Failed to save graph image")

    except Exception as e:
        logger.error(f"Failed to save graph image for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save graph image: {str(e)}")


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