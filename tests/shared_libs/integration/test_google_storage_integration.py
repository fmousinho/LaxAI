"""Integration tests for Google Cloud Storage client interactions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final
from uuid import uuid4

import numpy as np
import pytest

from shared_libs.common.google_storage import GCSPaths, GoogleStorageClient

pytestmark = pytest.mark.integration


def test_google_storage_download_variants(tmp_path: Path) -> None:
    """Exercise all download helpers across multiple blob types."""
    pytest.importorskip("cv2")

    tenant_id: Final[str] = os.environ.get("LAXAI_INTEGRATION_TENANT", "test-tenant")
    client = GoogleStorageClient(tenant_id)

    if not client._ensure_authenticated():
        pytest.skip("Unable to authenticate with Google Cloud Storage")

    paths = GCSPaths()

    video_id = f"pytest-{uuid4().hex}"
    process_prefix = paths.get_path("process_folder", video_id=video_id)
    assert process_prefix is not None

    def build_blob(name: str) -> str:
        return f"{process_prefix}{name}"

    blobs_to_cleanup: list[str] = []

    json_blob = build_blob("metrics.json")
    text_blob = build_blob("notes.txt")
    image_blob = build_blob("thumbnail.jpg")
    video_blob = build_blob("clip.mp4")

    # JSON payload
    payload = {"status": "ok", "video_id": video_id}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    # Text payload
    text_payload = f"Test run for {video_id}"

    # Image payload
    image_payload = np.full((12, 12, 3), 200, dtype=np.uint8)

    # Video payload (synthetic bytes)
    video_payload = b"mp4" + uuid4().bytes
    video_file_path = tmp_path / "clip_source.mp4"
    video_file_path.write_bytes(video_payload)

    try:
        assert client.upload_from_bytes(json_blob, payload_bytes)
        blobs_to_cleanup.append(json_blob)

        assert client.upload_from_string(text_blob, text_payload)
        blobs_to_cleanup.append(text_blob)

        assert client.upload_from_bytes(image_blob, image_payload)  # type: ignore[arg-type]
        blobs_to_cleanup.append(image_blob)

        assert client.upload_from_file(video_blob, str(video_file_path))
        blobs_to_cleanup.append(video_blob)

        # JSON download variants
        downloaded_json_str = client.download_as_string(json_blob)
        assert downloaded_json_str is not None
        assert json.loads(downloaded_json_str)["video_id"] == video_id

        downloaded_json_any = client.download_as_appropriate_type(json_blob)
        assert isinstance(downloaded_json_any, str)
        assert json.loads(downloaded_json_any)["status"] == "ok"

        # Text download
        downloaded_text = client.download_as_appropriate_type(text_blob)
        assert downloaded_text == text_payload

        # Image download
        downloaded_image = client.download_as_appropriate_type(image_blob)
        assert isinstance(downloaded_image, np.ndarray)
        assert downloaded_image.shape == image_payload.shape
        assert downloaded_image.dtype == np.uint8
        assert np.allclose(downloaded_image, image_payload, atol=5)

        # Video download
        downloaded_video = client.download_as_appropriate_type(video_blob)
        assert isinstance(downloaded_video, (bytes, bytearray))
        assert downloaded_video == video_payload

        # Download blob to disk
        download_path = tmp_path / "metrics.json"
        assert client.download_blob(json_blob, str(download_path))
        assert json.loads(download_path.read_text()) == payload

        # Ensure list includes our blobs (without tenant prefix)
        listed_blobs = client.list_blobs(prefix=process_prefix, include_user_id=False)
        assert json_blob in listed_blobs
        assert text_blob in listed_blobs
        assert image_blob in listed_blobs
        assert video_blob in listed_blobs

    finally:
        for blob_name in blobs_to_cleanup:
            client.delete_blob(blob_name)
