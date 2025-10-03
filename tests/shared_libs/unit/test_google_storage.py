from textwrap import dedent
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock

from shared_libs.common.google_storage import GCSPaths, GoogleStorageClient
from shared_libs.config.all_config import google_storage_config


@pytest.fixture
def temporary_gcs_paths_file(tmp_path, monkeypatch):
    """Create a temporary GCS paths YAML and point configuration to it."""

    yaml_path = tmp_path / "gcs_paths.yaml"
    yaml_path.write_text(
        dedent(
            """
            gcs:
              data_prefixes:
                raw_data: "raw/"
                process_root: "process/"
                process_folder: "process/{video_id}/"
                detections_path: "process/{video_id}/detections.json"
                custom_path: "runs/{run_id}/artifacts/{name}"
            """
        ).strip()
    )

    monkeypatch.setattr(google_storage_config, "gcs_paths_file", str(yaml_path))
    return yaml_path


class TestGCSPaths:
    def test_get_path_success(self, temporary_gcs_paths_file):
        paths = GCSPaths()
        result = paths.get_path("custom_path", run_id="run123", name="metrics.json")
        assert result == "runs/run123/artifacts/metrics.json"

    def test_get_path_missing_key(self, temporary_gcs_paths_file):
        paths = GCSPaths()
        with pytest.raises(KeyError):
            paths.get_path("unknown_key")

    def test_get_path_invalid_characters(self, temporary_gcs_paths_file):
        paths = GCSPaths()
        # Slash is disallowed in parameters
        result = paths.get_path("process_folder", video_id="foo/bar")
        assert result is None


class TestGoogleStorageClientDecorators:
    @pytest.fixture
    def client_with_bucket(self, monkeypatch):
        client = GoogleStorageClient("unit-test")
        client._authenticated = True

        mock_bucket = MagicMock()
        client._bucket = mock_bucket

        def fake_authenticate():
            client._authenticated = True
            return True

        monkeypatch.setattr(client, "_authenticate", fake_authenticate)
        return client, mock_bucket

    def test_list_blobs_builds_prefix_and_strips_results(self, client_with_bucket):
        client, mock_bucket = client_with_bucket
        mock_bucket.list_blobs.return_value = [
            SimpleNamespace(name="unit-test/process/sample/detections.json")
        ]

        results = client.list_blobs(
            prefix="process/sample",
            include_user_id=False,
            exclude_prefix_in_return=True,
        )

        assert mock_bucket.list_blobs.call_args.kwargs["prefix"] == "unit-test/process/sample/"
        assert results == {"detections.json"}
        # user_id should have trailing slash cached for future calls
        assert client.user_id.endswith("/")

    def test_upload_from_bytes_sets_content_type(self, client_with_bucket):
        client, mock_bucket = client_with_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        success = client.upload_from_bytes("process/sample/detections.json", b"{}")

        assert success is True
        mock_bucket.blob.assert_called_once_with("unit-test/process/sample/detections.json")
        mock_blob.upload_from_string.assert_called_once_with(b"{}", content_type="application/json")

    def test_upload_from_bytes_rejects_unknown_extension(self, client_with_bucket):
        client, mock_bucket = client_with_bucket
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        success = client.upload_from_bytes("process/sample/data.bin", b"binary")

        assert success is False
        mock_bucket.blob.assert_called_once_with("unit-test/process/sample/data.bin")
        mock_blob.upload_from_string.assert_not_called()

    def test_ensure_ready_returns_empty_set(self):
        client = GoogleStorageClient("unit-test")

        # Force authentication failure
        client._authenticated = False

        def fake_auth():
            return False

        client._authenticate = fake_auth  # type: ignore[assignment]

        result = client.list_blobs(prefix="process/sample")
        assert result == set()