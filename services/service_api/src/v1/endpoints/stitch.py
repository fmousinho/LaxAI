"""
Stitch proxy endpoints for LaxAI API Service.

Proxies requests to service_stitch FastAPI service.
"""

import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import google.auth.transport.requests
import google.oauth2.id_token
import httpx
import requests
from fastapi import APIRouter, HTTPException, Request, Response

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Handle application startup and shutdown events."""
    # Startup logic (if needed)
    yield
    # Shutdown logic
    await stitch_client.close()


router = APIRouter(prefix="/stitcher", tags=["Stitcher"], lifespan=lifespan)


class ServiceHTTPClient:
    """Base HTTP client for communicating with LaxAI microservices with token caching."""

    def __init__(self, service_name: str, service_url: str):
        self.base_url = service_url
        self._target_audience = service_url  # Required for service-to-service authentication
        # Increase timeout to 120 seconds to account for cold starts and artifact downloads
        self.client = httpx.AsyncClient(base_url=service_url, timeout=120.0)

        # Token caching
        self._cached_token: Optional[str] = None
        self._token_expiry: Optional[float] = None
        
        logger.info(
            "Initialized %s client | Base URL: %s | Target audience: %s",
            service_name, self.base_url, self._target_audience
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _get_cached_token(self) -> str:
        """Get a cached ID token, refreshing if necessary."""
        current_time = time.time()

        # Check if we have a valid cached token
        if self._cached_token and self._token_expiry and current_time < self._token_expiry:
            logger.debug("Using cached token (expires in %.0f seconds)", self._token_expiry - current_time)
            return self._cached_token

        # Fetch new token from metadata server (official method for Cloud Run)
        if self._target_audience:
            try:
                logger.info("Fetching new ID token for audience: %s", self._target_audience)
                
                # Official Cloud Run service-to-service authentication method
                metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
                headers = {"Metadata-Flavor": "Google"}
                params = {"audience": self._target_audience}
                
                response = requests.get(metadata_url, headers=headers, params=params, timeout=5)
                response.raise_for_status()
                token = response.text
                
                if token:
                    self._cached_token = token
                    # ID tokens typically expire in 1 hour (3600 seconds)
                    # We'll refresh 5 minutes early to be safe
                    self._token_expiry = current_time + 3300  # 55 minutes
                    
                    # Decode token to log claims for debugging
                    import base64
                    import json
                    try:
                        parts = token.split('.')
                        if len(parts) == 3:
                            payload = parts[1]
                            payload += '=' * (4 - len(payload) % 4)
                            decoded = json.loads(base64.urlsafe_b64decode(payload))
                            logger.info(
                                "Successfully fetched ID token | Length: %d | aud: %s | iss: %s | email: %s",
                                len(token), 
                                decoded.get('aud'), 
                                decoded.get('iss'), 
                                decoded.get('email', 'N/A')
                            )
                    except Exception as decode_err:
                        logger.warning("Could not decode token for logging: %s", decode_err)
                        logger.info("Successfully fetched ID token from metadata server (length: %d)", len(token))
                    
                    return token
                else:
                    logger.error("Metadata server returned empty token for audience: %s", self._target_audience)
            except Exception as e:
                logger.error("Error fetching ID token from metadata server for audience %s: %s\n%s", 
                           self._target_audience, e, traceback.format_exc())
        else:
            logger.error("No target audience configured for token generation")

        return ""

class StitchClient(ServiceHTTPClient):
    """HTTP client for communicating with service_stitch."""

    def __init__(self):
        super().__init__(
            service_name="laxai-service-stitcher",
            service_url="https://laxai-service-stitcher-kfccnooita-uc.a.run.app"
        )

# Global client instance
stitch_client = StitchClient()


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_to_stitch(path: str, request: Request):
    """
    Catch-all proxy endpoint that forwards all requests to the stitch service.
    Automatically adds authentication headers using cached tokens.
    """
    url = f"{path}"
    try:
        token = stitch_client._get_cached_token()
        
        # Build headers - copy all from request EXCEPT Authorization and Host
        # The host header must match the target service, not the incoming request
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("authorization", "host")}
        
        # Set the correct host header for the target service
        headers["host"] = "laxai-service-stitcher-kfccnooita-uc.a.run.app"
        
        # Always set outgoing Authorization header to service account token
        if not token:
            logger.error(
                "No valid token available | Method: %s | URL: %s | Base URL: %s",
                request.method, url, stitch_client.base_url
            )
        else:
            headers["Authorization"] = f"Bearer {token}"
        
        if "content-type" not in [h.lower() for h in headers.keys()]:
            headers["Content-Type"] = "application/json"
        
        body = await request.body()
        
        # Log detailed request info for debugging auth issues
        logger.info(
            "Proxying request | Method: %s | URL: %s | Has token: %s | Token prefix: %s",
            request.method, 
            url, 
            bool(token),
            token[:50] + "..." if token else "None"
        )
        logger.info(
            "Outgoing headers: %s",
            {k: (v[:50] + "..." if len(v) > 50 else v) for k, v in headers.items()}
        )
        
        response = await stitch_client.client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            params=request.query_params,
        )
        
        logger.info(
            "Proxy response received | Status: %s | URL: %s | Content-Length: %s",
            response.status_code, 
            url,
            response.headers.get('content-length', 'N/A')
        )
        
        # Log additional details for auth failures
        if response.status_code == 401:
            logger.error(
                "Authentication failed (401) | URL: %s | Response: %s",
                url,
                response.text[:500] if response.text else "No response body"
            )
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    except httpx.HTTPStatusError as e:
        logger.error(
            "HTTP error proxying to stitch service | URL: %s | Status: %s | Response: %s\n%s",
            url,
            e.response.status_code if e.response else "N/A",
            e.response.text if e.response else "N/A",
            traceback.format_exc()
        )
        return Response(
            content=e.response.content if e.response else b"",
            status_code=e.response.status_code if e.response else 500,
            headers=dict(e.response.headers) if e.response else {},
        )
    except Exception as e:
        logger.error(
            "Error proxying request to stitch service | URL: %s | Exception: %s\n%s",
            url, str(e), traceback.format_exc()
        )
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {str(e)}")