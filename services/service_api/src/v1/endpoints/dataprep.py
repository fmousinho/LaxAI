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
        # Use Google Cloud service-to-service discovery as primary method
        # Cloud Run services can communicate using the service URL
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", None)
        if not project_id:
            dataprep_service_url = f"http://service_dataprep.{project_id}.internal:8080"
        else:
            external_url = f"https://laxai-service-dataprep.a.run.app:8080"
            dataprep_service_url = os.getenv("SERVICE_DATAPREP_URL", external_url)

        self.base_url = dataprep_service_url
        self.client = httpx.AsyncClient(base_url=dataprep_service_url, timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _proxy_request(self, method: str, path: str, **kwargs) -> dict:
        """Proxy a request to service_dataprep."""
        try:
            # Fetch Google Cloud ID token for authentication
            auth_req = google.auth.transport.requests.Request()
            id_token = google.oauth2.id_token.fetch_id_token(auth_req, self.base_url)
            
            # Prepare headers with Bearer token and content type
            headers = {
                "Authorization": f"Bearer {id_token}",
                "Content-Type": "application/json"
            }
            
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
            logger.error("Request error to service_dataprep: %s", e)
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
    data = await dataprep_client._proxy_request("GET", "/process-folders", params={"tenant_id": tenant_id})
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
        json={"process_folder": request.process_folder},
        params={"tenant_id": tenant_id}
    )
    return StartPrepResponse(**data)


@router.get("/verification-images", response_model=VerificationImagesResponse)
async def get_verification_images(tenant_id: str) -> VerificationImagesResponse:
    """
    Get images for verification.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request("GET", "/verification-images", params={"tenant_id": tenant_id})
    return VerificationImagesResponse(**data)


@router.post("/record-response", response_model=RecordResponseResponse)
async def record_response(request: RecordResponseRequest, tenant_id: str) -> RecordResponseResponse:
    """
    Record a user response for verification.

    Proxies to service_dataprep.
    """
    data = await dataprep_client._proxy_request(
        "POST",
        "/record-response",
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