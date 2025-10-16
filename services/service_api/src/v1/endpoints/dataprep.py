"""
DataPrep proxy endpoints for LaxAI API Service.

Proxies requests to service_dataprep FastAPI service.
"""

import logging
import os
import time
from typing import Optional

import google.auth.transport.requests
import google.oauth2.id_token
import httpx
from fastapi import APIRouter, HTTPException

from ..schemas.dataprep import (
    ErrorResponse,
    GraphStatisticsResponse,
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

router = APIRouter(prefix="/dataprep", tags=["DataPrep"])


class ServiceHTTPClient:
    """Base HTTP client for communicating with LaxAI microservices with token caching."""

    def __init__(self, service_name: str, default_url: str, service_url_env_var: str):
        # Allow explicit override via environment variable (for local/dev environments)
        explicit_url = os.getenv(service_url_env_var)

        if explicit_url:
            service_url = explicit_url.rstrip("/")
            self._target_audience = service_url  # Use ID token for external URLs
        else:
            # Use external Cloud Run service URL with proper authentication
            service_url = default_url
            self._target_audience = service_url  # Required for external service authentication

        self.base_url = service_url
        self.client = httpx.AsyncClient(base_url=service_url, timeout=30.0)

        # Token caching
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[float] = None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _get_cached_token(self) -> str:
        """Get a cached ID token, refreshing if necessary."""
        current_time = time.time()

        # Check if we have a valid cached token
        if self._cached_token and self._token_expiry and current_time < self._token_expiry:
            return self._cached_token

        # Fetch new token
        if self._target_audience:
            auth_req = google.auth.transport.requests.Request()
            token = google.oauth2.id_token.fetch_id_token(auth_req, self._target_audience)
            if token:
                self._cached_token = token
                # ID tokens typically expire in 1 hour (3600 seconds)
                # We'll refresh 5 minutes early to be safe
                self._token_expiry = current_time + 3300  # 55 minutes
                return token

        return ""

    async def _proxy_request(self, method: str, path: str, service_prefix: str, **kwargs) -> dict:
        """Proxy a request to the target service."""
        try:
            headers = {"Content-Type": "application/json"}

            # Get cached token instead of fetching new one
            token = self._get_cached_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

            # Merge with any existing headers from kwargs (if provided)
            if "headers" in kwargs:
                headers.update(kwargs["headers"])
            kwargs["headers"] = headers

            url = f"/api/v1/{service_prefix}{path}"
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error from %s service: %s", service_prefix, e)
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            logger.error("Request error to %s service (base_url=%s): %s", service_prefix, self.base_url, repr(e))
            raise HTTPException(status_code=503, detail="Service unavailable")


class DataPrepClient(ServiceHTTPClient):
    """HTTP client for communicating with service_dataprep."""

    def __init__(self):
        super().__init__(
            service_name="laxai-service-dataprep",
            default_url="https://laxai-service-dataprep-kfccnooita-uc.a.run.app",
            service_url_env_var="SERVICE_DATAPREP_URL"
        )

    async def _proxy_request(self, method: str, path: str, **kwargs) -> dict:
        """Proxy a request to service_dataprep."""
        return await super()._proxy_request(method, path, "dataprep", **kwargs)


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
        json={"pair_id": request.pair_id, "decision": request.decision},
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


@router.get("/graph-statistics", response_model=GraphStatisticsResponse)
async def get_graph_statistics(tenant_id: str) -> GraphStatisticsResponse:
    """
    Get statistics about the current graph state.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request("GET", "/graph-statistics", params={"tenant_id": tenant_id})
    return GraphStatisticsResponse(**data)


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