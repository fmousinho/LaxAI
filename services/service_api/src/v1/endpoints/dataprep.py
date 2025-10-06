"""
DataPrep proxy endpoints for LaxAI API Service.

Proxies requests to service_dataprep FastAPI service.
"""

import logging
import os
from typing import List

import google.auth.transport.requests
import google.oauth2.id_token
import httpx
from fastapi import APIRouter, HTTPException, Request

from ..schemas.dataprep import (
    ErrorResponse,
    ProcessFoldersResponse,
    RecordResponseRequest,
    RecordResponseResponse,
    SaveGraphImageResponse,
    SaveGraphResponse,
    SplitTrackRequest,
    SplitTrackResponse,
    StartPrepRequest,
    StartPrepResponse,
    SuspendPrepResponse,
    VerificationImagesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dataprep", tags=["dataprep"])


class DataPrepClient:
    """HTTP client for communicating with service_dataprep."""

    def __init__(self):
        service_name = os.getenv("SERVICE_DATAPREP_NAME", "laxai-service-dataprep")
        port = os.getenv("SERVICE_DATAPREP_PORT", "8080")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "laxai-466119")
        region = "us-central1"  # Cloud Run region

        # Allow explicit override via environment variable (for local/dev environments)
        explicit_url = os.getenv("SERVICE_DATAPREP_URL")

        if explicit_url:
            dataprep_service_url = explicit_url.rstrip("/")
            self._target_audience = dataprep_service_url  # Use ID token for external URLs
        else:
            # Use external Cloud Run service URL with proper authentication
            dataprep_service_url = "https://laxai-service-dataprep-kfccnooita-uc.a.run.app"
            self._target_audience = dataprep_service_url  # Required for external service authentication

        self.base_url = dataprep_service_url
        self.client = httpx.AsyncClient(base_url=dataprep_service_url, timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _proxy_request(self, method: str, path: str, **kwargs) -> dict:
        """Proxy a request to service_dataprep."""
        try:
            headers = {"Content-Type": "application/json"}

            if self._target_audience:
                auth_req = google.auth.transport.requests.Request()
                id_token = google.oauth2.id_token.fetch_id_token(auth_req, self._target_audience)
                headers["Authorization"] = f"Bearer {id_token}"
            
            # Merge with any existing headers from kwargs (if provided)
            if "headers" in kwargs:
                headers.update(kwargs["headers"])
            kwargs["headers"] = headers
            
            url = f"/api/v1/dataprep{path}"
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error from service_dataprep: %s", e)
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            logger.error("Request error to service_dataprep (base_url=%s): %s", self.base_url, repr(e))
            raise HTTPException(status_code=503, detail="Service unavailable")


# Global client instance
dataprep_client = DataPrepClient()


@router.on_event("shutdown")
async def shutdown_event():
    """Shutdown event to close the HTTP client."""
    await dataprep_client.close()


@router.get("/process-folders", response_model=ProcessFoldersResponse)
async def get_process_folders(tenant_id: str) -> ProcessFoldersResponse:
    """
    Get list of available process folders for a tenant.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request("GET", "/folders", params={"tenant_id": tenant_id})
    return ProcessFoldersResponse(**data)


@router.post("/start", response_model=StartPrepResponse)
async def start_prep(request: StartPrepRequest, tenant_id: str) -> StartPrepResponse:
    """
    Start a verification session for a process folder.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request(
        "POST",
        "/start",
        json={"video_id": request.video_id},
        params={"tenant_id": tenant_id}
    )
    return StartPrepResponse(**data)


@router.get("/verification-images", response_model=VerificationImagesResponse)
async def get_verification_images(tenant_id: str) -> VerificationImagesResponse:
    """
    Get images for verification.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request("GET", "/verify", params={"tenant_id": tenant_id})
    return VerificationImagesResponse(**data)


@router.post("/record-response", response_model=RecordResponseResponse)
async def record_response(request: RecordResponseRequest, tenant_id: str) -> RecordResponseResponse:
    """
    Record a user response for verification.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request(
        "POST",
        "/respond",
        json={"decision": request.decision},
        params={"tenant_id": tenant_id}
    )
    return RecordResponseResponse(**data)


@router.post("/save-graph", response_model=SaveGraphResponse)
async def save_graph(tenant_id: str) -> SaveGraphResponse:
    """
    Save the current graph state.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request("POST", "/save-graph", params={"tenant_id": tenant_id})
    return SaveGraphResponse(**data)


@router.post("/save-graph-image", response_model=SaveGraphImageResponse)
async def save_graph_image(tenant_id: str) -> SaveGraphImageResponse:
    """
    Save the current graph as an image.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request("POST", "/save-graph-image", params={"tenant_id": tenant_id})
    return SaveGraphImageResponse(**data)


@router.post("/suspend", response_model=SuspendPrepResponse)
async def suspend_prep(tenant_id: str) -> SuspendPrepResponse:
    """
    Suspend the current verification session.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request("POST", "/suspend", params={"tenant_id": tenant_id})
    return SuspendPrepResponse(**data)


@router.post("/split-track", response_model=SplitTrackResponse)
async def split_track_at_frame(request: SplitTrackRequest, tenant_id: str) -> SplitTrackResponse:
    """
    Split a track that was incorrectly merged by the tracker.

    This endpoint allows correcting cases where the tracker incorrectly grouped
    two different players in the same track. The track is split at the frame
    where the player switch occurs.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request(
        "POST",
        "/split-track",
        json={"track_id": request.track_id, "crop_image_name": request.crop_image_name},
        params={"tenant_id": tenant_id}
    )
    return SplitTrackResponse(**data)