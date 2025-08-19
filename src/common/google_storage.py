"""
Google Cloud Storage utilities for the LaxAI project.

This module provides a client and helpers for interacting with Google Cloud Storage,
including error handling, credential management, and common operations.
 
 Cloud Run vs Local development (concise):
 - Cloud Run: rely on Application Default Credentials (ADC) provided by the runtime.
     Do not set GOOGLE_APPLICATION_CREDENTIALS. Ensure the Cloud Run service account
     has the minimum IAM permissions for the target bucket (for example,
     roles/storage.objectAdmin or a narrower custom role).
 - Local/dev: either set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON
     file or run `gcloud auth application-default login` to populate ADC. This
     module also supports providing credentials via an env var containing a JSON
     string or a path configured by `google_storage_config.credentials_name`.
 - Security: prefer workload identity / attached service accounts on GCP and
     restrict service-account keys when used.
"""
#==========================================================================
# Google Cloud Storage Client with Error Handling and Common Operations
#
# This module is hardcoded to a specific project and bucket for simplicity.
#
#
# TODO:
# - Add support for multi-tenancy
#==========================================================================
from __future__ import annotations

"""Google Cloud Storage helpers.

This module is import-safe: heavy optional dependencies (cv2, numpy, supervision,
google-cloud) are only required at runtime when their features are used.

The authentication logic prefers, in order:
  1. Explicit credentials passed to GoogleStorageClient
  2. An environment variable configured by `google_storage_config.credentials_name`
      which may contain a path to a JSON service account file or the JSON string
      itself
  3. Application Default Credentials (ADC) — this is the recommended mode for
      Cloud Run, GCE, and GKE where the runtime provides credentials.

The client will log helpful messages when authentication fails.
"""

import logging
import os
import tempfile
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Optional imports are deferred to runtime when needed
try:
    import yaml
except Exception:  # pragma: no cover - optional
    yaml = None

try:
    # google-cloud imports may not be installed in minimal environments
    from google.auth.exceptions import DefaultCredentialsError  # type: ignore
    from google.cloud import storage  # type: ignore
    from google.oauth2 import service_account  # type: ignore
except Exception:  # pragma: no cover - optional
    storage = None
    service_account = None
    DefaultCredentialsError = Exception

# Try to load optional local config; callers can also pass config explicitly
try:
    from config.all_config import google_storage_config  # type: ignore
except Exception:  # pragma: no cover - optional
    google_storage_config = None


class GCSPaths:
    """Load path templates for GCS from a YAML file (optional).

    If PyYAML is not installed or the file is missing the loader will return an
    empty mapping and the class will still be usable.
    """

    def __init__(self, gcs_paths_file: Optional[str] = None):
        self.gcs_paths_file = gcs_paths_file or (getattr(google_storage_config, "gcs_paths_file", None) if google_storage_config else None)
        self.paths: Dict[str, str] = self._load_paths() if self.gcs_paths_file else {}

    def _load_paths(self) -> Dict[str, str]:
        if yaml is None:
            logger.debug("PyYAML not installed; skipping GCS paths load")
            return {}
        path = os.path.abspath(self.gcs_paths_file)
        if not os.path.exists(path):
            logger.debug("GCS paths file not found: %s", path)
            return {}
        try:
            with open(path, "r", encoding="utf8") as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get("gcs", {}).get("data_prefixes", {}) or {}
        except Exception:
            logger.exception("Failed to load GCS paths file %s", path)
            return {}

    def get_path(self, key: str, **kwargs) -> Optional[str]:
        template = self.paths.get(key)
        if not template:
            logger.debug("GCS path key not found: %s", key)
            return None
        try:
            # Basic validation: path parameters shouldn't include path separators
            for k, v in kwargs.items():
                if isinstance(v, str) and any(ch in v for ch in ("/", "\\", ".")):
                    logger.error("Invalid character in path param %s", k)
                    return None
            return template.format(**kwargs)
        except Exception:
            logger.exception("Failed to format GCS path for key %s", key)
            return None


def ensure_ready(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self._ensure_authenticated():
            logger.error("GCS client not authenticated")
            if func.__name__.startswith(("download", "get")):
                return None
            if func.__name__ == "list_blobs":
                return set()
            return False
        if not getattr(self, "_bucket", None):
            logger.error("GCS bucket not available")
            if func.__name__ == "list_blobs":
                return set()
            return False
        return func(self, *args, **kwargs)

    return wrapper


def normalize_user_path(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "user_id", None) and not str(self.user_id).endswith("/"):
            self.user_id = f"{self.user_id}/"
        return func(self, *args, **kwargs)

    return wrapper


def build_full_path(path_param_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            path_value = bound.arguments.get(path_param_name)
            if path_value and getattr(self, "user_id", None):
                if not str(path_value).startswith(self.user_id):
                    bound.arguments[path_param_name] = f"{self.user_id}{path_value}"
                    call_kwargs = {k: v for k, v in bound.arguments.items() if k != "self"}
                    return func(self, **call_kwargs)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class GoogleStorageClient:
    """Thin GCS client with lazy auth and ADC-friendly behavior.

    Parameters
    ----------
    tenant_id:
        A per-tenant prefix that will be applied to object keys when using the
        decorated methods.
    credentials:
        Optional explicit credentials object.
    config:
        Optional config object providing attributes: project_id, bucket_name,
        credentials_name (env var key). If not provided the module-level
        google_storage_config will be used if available.
    """

    def __init__(self, tenant_id: str, credentials: Optional[Any] = None, config: Optional[Any] = None):
        self.config = config or google_storage_config
        if self.config is None:
            raise RuntimeError("google_storage_config is required either via import or passed to GoogleStorageClient")
        self.user_id = tenant_id
        self.credentials = credentials
        self._client = None
        self._bucket = None
        self._authenticated = False

    def _authenticate(self) -> bool:
        if storage is None:
            logger.error("google-cloud-storage is not installed")
            return False

        try:
            # 1) Explicit credentials object
            if self.credentials is not None:
                logger.info("GCS auth: using explicit credentials object passed to client")
                self._client = storage.Client(credentials=self.credentials, project=getattr(self.config, "project_id", None))

            else:
                # 2) Environment variable configured in config (may be path or JSON)
                cred_env_key = getattr(self.config, "credentials_name", "GOOGLE_APPLICATION_CREDENTIALS")
                cred_env = os.environ.get(cred_env_key)

                if cred_env:
                    # If it's a filesystem path, load from file
                    if os.path.exists(cred_env):
                        logger.info("GCS auth: using service account JSON file from env %s", cred_env_key)
                        proj = getattr(self.config, "project_id", None)
                        if proj:
                            self._client = storage.Client.from_service_account_json(cred_env, project=proj)
                        else:
                            self._client = storage.Client.from_service_account_json(cred_env)
                    else:
                        # Try parsing JSON from the env var
                        try:
                            import json as _json

                            info = _json.loads(cred_env)
                            if service_account is not None:
                                creds = service_account.Credentials.from_service_account_info(info)
                                self._client = storage.Client(credentials=creds, project=getattr(self.config, "project_id", None))
                                logger.info("GCS auth: created credentials from JSON string in env var %s", cred_env_key)
                            else:
                                logger.warning("service_account helper not available; falling back to ADC")
                        except Exception:
                            logger.warning("Env var %s present but not a file or valid JSON; will try ADC", cred_env_key)

                # 3) Attempt ADC if client still not constructed
                if getattr(self, "_client", None) is None:
                    logger.info("GCS auth: attempting Application Default Credentials (ADC)")
                    proj = getattr(self.config, "project_id", None)
                    try:
                        if proj:
                            self._client = storage.Client(project=proj)
                        else:
                            self._client = storage.Client()
                        logger.info("GCS auth: ADC succeeded")
                    except Exception:
                        logger.debug("GCS ADC attempt failed; will try metadata service as fallback")
                        self._client = None

                    # If ADC didn't return a client, try metadata-based client construction
                    if getattr(self, "_client", None) is None:
                        try:
                            import requests
                            response = requests.get(
                                'http://metadata.google.internal/computeMetadata/v1/project/project-id',
                                headers={'Metadata-Flavor': 'Google'},
                                timeout=2
                            )
                            if response.status_code == 200:
                                logger.info("GCS auth: metadata service reachable; relying on platform-provided credentials")
                                # Construct client without explicit creds (ADC should still work)
                                proj = getattr(self.config, "project_id", None) or response.text
                                self._client = storage.Client(project=proj) if storage else None
                        except Exception:
                            logger.debug("Metadata service call failed or not reachable")

            if not getattr(self, "_client", None):
                logger.error("Unable to create google-cloud-storage client; no credentials available")
                return False

            # Verify bucket and mark authenticated
            self._bucket = self._client.bucket(getattr(self.config, "bucket_name"))
            self._bucket.reload()
            self._authenticated = True
            logger.info("Authenticated with GCS bucket %s", getattr(self.config, "bucket_name"))
            return True

        except DefaultCredentialsError as e:
            logger.error("GCS authentication failed (DefaultCredentialsError): %s", e)
            logger.error("If running on Cloud Run, ensure the service has an attached service account with access to the bucket.")
            return False
        except Exception:
            logger.exception("Unexpected error during GCS authentication")
            return False

    def _ensure_authenticated(self) -> bool:
        if not self._authenticated:
            return self._authenticate()
        return True

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_client"] = None
        s["_bucket"] = None
        s["_authenticated"] = False
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)

    @ensure_ready
    @normalize_user_path
    @build_full_path("prefix")
    def list_blobs(self, prefix: Optional[str] = None, include_user_id: bool = True, delimiter: Optional[str] = None, exclude_prefix_in_return: bool = False) -> Set[str]:
        try:
            if prefix:
                prefix = prefix.rstrip("/") + "/"
            iterator = self._bucket.list_blobs(prefix=prefix, delimiter=delimiter)  # type: ignore
            uid_len = len(self.user_id) if self.user_id else 0
            result: Set[str] = set()
            if delimiter:
                list(iterator)
                for p in iterator.prefixes:
                    if exclude_prefix_in_return and prefix and p.startswith(prefix):
                        result.add(p[len(prefix):])
                    elif include_user_id or uid_len == 0:
                        result.add(p)
                    else:
                        result.add(p[uid_len:])
            else:
                for blob in iterator:
                    if exclude_prefix_in_return and prefix and blob.name.startswith(prefix):
                        result.add(blob.name[len(prefix):])
                    elif include_user_id or uid_len == 0:
                        result.add(blob.name)
                    else:
                        result.add(blob.name[uid_len:])
            return result
        except Exception:
            logger.exception("Failed to list blobs")
            raise

    @ensure_ready
    @normalize_user_path
    @build_full_path("destination_blob_name")
    def upload_from_bytes(self, destination_blob_name: str, data: bytes, content_type: Optional[str] = None) -> bool:
        try:
            blob = self._bucket.blob(destination_blob_name)  # type: ignore
            # Image handling is optional and requires cv2 and numpy
            if destination_blob_name.endswith((".jpg", ".jpeg")):
                try:
                    import cv2  # type: ignore
                    import numpy as np  # type: ignore
                except Exception:
                    raise RuntimeError("cv2 and numpy are required to process image bytes; install opencv-python-headless and numpy")
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    logger.error("Failed to decode image bytes")
                    return False
                _, enc = cv2.imencode(".jpg", img)
                blob.upload_from_string(enc.tobytes(), content_type=content_type or "image/jpeg")
                return True

            # JSON and raw uploads
            if destination_blob_name.endswith(".json"):
                import json as _json

                blob.upload_from_string(_json.dumps(data), content_type=content_type or "application/json")
                return True

            # Fallback for raw bytes
            blob.upload_from_string(data, content_type=content_type or "application/octet-stream")
            return True
        except Exception:
            logger.exception("upload_from_bytes failed")
            return False

    @ensure_ready
    @normalize_user_path
    @build_full_path("destination_blob_name")
    def upload_from_string(self, destination_blob_name: str, data: str, content_type: Optional[str] = None) -> bool:
        """
        Upload a string (text or JSON) to the specified blob path.

        This exists for convenience because some callers prepare JSON/text
        content as Python strings. Binary/image uploads should use
        `upload_from_bytes`.
        """
        try:
            blob = self._bucket.blob(destination_blob_name)  # type: ignore

            # If the destination looks like JSON prefer the JSON content type
            if destination_blob_name.endswith('.json'):
                blob.upload_from_string(data, content_type=content_type or "application/json")
                return True

            # Default to plain text
            blob.upload_from_string(data, content_type=content_type or "text/plain")
            return True
        except Exception:
            logger.exception("upload_from_string failed")
            return False

    @ensure_ready
    @normalize_user_path
    @build_full_path("source_blob_name")
    def download_as_appropriate_type(self, source_blob_name: str) -> Optional[Any]:
        try:
            blob = self._bucket.blob(source_blob_name)  # type: ignore
            content = blob.download_as_bytes()
            if source_blob_name.endswith((".jpg", ".jpeg")):
                try:
                    import cv2  # type: ignore
                    import numpy as np  # type: ignore
                except Exception:
                    raise RuntimeError("cv2 and numpy are required to process images")
                img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return None
            if source_blob_name.endswith(('.json', '.txt')):
                return content.decode('utf-8')
            # For video or unknown types return raw bytes
            return content
        except Exception:
            logger.exception("Failed to download blob as appropriate type")
            return None


def get_storage(*args, **kwargs) -> GoogleStorageClient:
    return GoogleStorageClient(*args, **kwargs)
