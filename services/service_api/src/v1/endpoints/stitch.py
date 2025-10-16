"""
Stitch proxy endpoints for LaxAI API Service.

Proxies requests to service_stitch FastAPI service.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import google.auth.transport.requests
import google.oauth2.id_token
import httpx
from fastapi import APIRouter, HTTPException, Request, Response

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Handle application startup and shutdown events."""
    # Startup logic (if needed)
    yield
    # Shutdown logic
    await stitch_client.close()


router = APIRouter(prefix="/stitch", tags=["Stitch"], lifespan=lifespan)


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

class StitchClient(ServiceHTTPClient):
    """HTTP client for communicating with service_stitch."""

    def __init__(self):
        super().__init__(
            service_name="laxai-service-stitch",
            default_url="https://laxai-service-stitch-kfccnooita-uc.a.run.app",
            service_url_env_var="SERVICE_STITCH_URL"
        )

    async def _proxy_request(self, method: str, path: str, **kwargs) -> dict:
        """Proxy a request to service_stitch."""
        return await super()._proxy_request(method, path, "stitch", **kwargs)


# Global client instance
stitch_client = StitchClient()


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_to_stitch(path: str, request: Request):
    """
    Catch-all proxy endpoint that forwards all requests to the stitch service.

    Automatically adds authentication headers using cached tokens.
    """
    try:
        # Get the authorization token
        token = stitch_client._get_cached_token()

        # Prepare headers - copy original headers and add auth
        headers = dict(request.headers)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Ensure Content-Type is set for JSON requests
        if "content-type" not in [h.lower() for h in headers.keys()]:
            headers["Content-Type"] = "application/json"

        # Get request body
        body = await request.body()

        # Forward the request to stitch service
        url = f"/api/v1/stitch/{path}"
        response = await stitch_client.client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            params=request.query_params,
        )

        # Return the response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error proxying to stitch service: %s", e)
        return Response(
            content=e.response.content,
            status_code=e.response.status_code,
            headers=dict(e.response.headers),
        )
    except Exception as e:
        logger.error("Error proxying request to stitch service: %s", e)
        raise HTTPException(status_code=500, detail="Internal proxy error")