"""
Service-specific conftest for tracking tests.
This conftest provides fixtures specific to the tracking service.
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add service src to path for imports
service_src = os.path.join(os.path.dirname(__file__), '..', 'src')
if service_src not in sys.path:
    sys.path.insert(0, service_src)

# Add shared_libs to path 
shared_libs = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared_libs')
if shared_libs not in sys.path:
    sys.path.insert(0, shared_libs)

# Global mock setup for all tests
@pytest.fixture(scope="session", autouse=True)
def mock_external_dependencies():
    """Mock external dependencies for all tests."""
    # Mock computer vision and ML libraries (but keep supervision, torch, torchvision and PIL real)
    sys.modules['cv2'] = Mock()
    # sys.modules['supervision'] = Mock()  # Keep supervision real for imports
    # sys.modules['torch'] = Mock()  # Keep torch real for torchvision compatibility
    # sys.modules['torchvision'] = Mock()  # Keep torchvision real for transforms
    # sys.modules['PIL'] = Mock()  # Keep PIL real for supervision compatibility
    # sys.modules['PIL.Image'] = Mock()  # Keep PIL real for supervision compatibility

    # Mock application-specific modules
    mock_modules = [
        'config.all_config',
        'config.training_config',
        'config.transforms',  # Mock transforms to avoid ColorJitter issues
        'modules.clustering_processor',
        'modules.crop_extractor_processor',
        'modules.emb_processor',
        'models.detection_model',
        'models.affine_tracker',
        'common.google_storage',
        'common.pipeline_step',
        'common.gcs_paths',
        'common.detection',
        'common.tracker',
        'common.pipeline',
        'common.track_to_player',
        'utils.id_generator'
    ]

    for module in mock_modules:
        sys.modules[module] = Mock()


@pytest.fixture
def mock_config():
    """Create a standard mock configuration for testing."""
    config = Mock()
    config.frames_per_video = 100
    config.num_workers = 2
    config.verbose = False
    config.save_intermediate = False
    
    # Add attributes that TrackGeneratorPipeline expects
    config.detection_config = Mock()
    config.detection_config.model_path = "mock_model_path"
    config.detection_config.confidence_threshold = 0.5
    config.detection_config.batch_size = 8
    
    # Mock training_config attributes that are accessed
    config.training_config = Mock()
    config.training_config.num_workers = 2
    
    return config


@pytest.fixture
def mock_storage_client():
    """Create a mock Google Cloud Storage client."""
    storage = Mock()
    storage.bucket_name = "test-bucket"
    storage.upload_from_string = Mock(return_value=True)
    storage.download_as_string = Mock(return_value=None)
    storage.blob_exists = Mock(return_value=False)
    storage.list_blobs = Mock(return_value=set())  # Return empty set of strings
    storage.delete_blob = Mock(return_value=True)

    # Mock video capture with default behavior
    mock_cap = Mock()
    mock_cap.isOpened = Mock(return_value=True)
    mock_cap.get = Mock(side_effect=lambda prop: {
        7: 1000,  # CAP_PROP_FRAME_COUNT
        1: 0,     # CAP_PROP_POS_FRAMES (current frame)
    }.get(prop, 0))
    mock_cap.read = Mock(return_value=(True, Mock()))  # (ret, frame)
    mock_cap.set = Mock(return_value=True)
    storage.get_video_capture = Mock(return_value=mock_cap)

    return storage


@pytest.fixture
def mock_detection_model():
    """Create a mock detection model."""
    model = Mock()
    # Mock Detections object
    mock_detections = Mock()
    mock_detections.__len__ = Mock(return_value=5)
    mock_detections.data = {'frame_index': list(range(5))}

    model.generate_detections = Mock(return_value=mock_detections)
    return model


@pytest.fixture
def mock_tracker():
    """Create a mock tracker."""
    tracker = Mock()
    tracker.get_identity_affine_matrix = Mock(return_value=Mock())
    tracker.calculate_affine_transform = Mock(return_value=Mock())
    tracker.update_with_transform = Mock(return_value=Mock())
    return tracker


@pytest.fixture
def mock_path_manager():
    """Create a mock GCS path manager."""
    path_manager = Mock()
    path_manager.get_path = Mock(side_effect=lambda *args, **kwargs: f"mock/path/{args[0] if args else 'default'}")
    return path_manager


@pytest.fixture
def mock_pipeline_step():
    """Create a mock PipelineStep."""
    step = Mock()
    step.description = "Mock pipeline step"
    step.name = "mock_step"
    return step


@pytest.fixture
def pipeline(mock_config, mock_storage_client, mock_detection_model, mock_tracker, mock_path_manager, mock_pipeline_step):
    """Create a fully configured TrackGeneratorPipeline for testing."""
    # Ensure the mock storage client returns proper values
    mock_storage_client.list_blobs.return_value = set()
    
    # Mock the imports and dependencies at the module level
    with patch('common.google_storage.get_storage', return_value=mock_storage_client), \
         patch('common.detection.DetectionModel', return_value=mock_detection_model), \
         patch('common.tracker.AffineAwareByteTrack', return_value=mock_tracker), \
         patch('common.google_storage.GCSPaths', return_value=mock_path_manager), \
         patch('config.all_config.training_config', mock_config.training_config), \
         patch('common.pipeline_step.PipelineStep', return_value=mock_pipeline_step), \
         patch('common.pipeline.Pipeline.save_checkpoint', return_value=None), \
         patch('common.pipeline.Pipeline.load_checkpoint', return_value=None):

        # Import the class within the patch context
        from unverified_track_generator_pipeline import TrackGeneratorPipeline
        
        pipeline = TrackGeneratorPipeline(mock_config, "test-tenant")
        return pipeline


@pytest.fixture
def sample_video_context():
    """Create sample video processing context."""
    return {
        "raw_video_path": "test_video.mp4",
        "video_guid": "test-video-guid",
        "video_blob_name": "tenant1/videos/test_video.mp4",
        "video_folder": "tenant1/processed/test-video-guid"
    }


@pytest.fixture
def sample_detection_results():
    """Create sample detection processing results."""
    return {
        "status": "completed",
        "all_detections": [
            Mock(data={'frame_index': [0, 1, 2]}),
            Mock(data={'frame_index': [15, 16, 17]})
        ],
        "total_detections": 6,
        "crop_paths": [
            "tenant1/unverified_tracks/test-video-guid/track_1/crop_0_0.jpg",
            "tenant1/unverified_tracks/test-video-guid/track_1/crop_1_15.jpg",
            "tenant1/unverified_tracks/test-video-guid/track_2/crop_0_30.jpg"
        ],
        "total_crops": 3
    }


@pytest.fixture
def sample_checkpoint_data():
    """Create sample checkpoint data."""
    return {
        "pipeline_name": "track_generator_pipeline",
        "run_guid": "test-run-guid",
        "run_folder": "process/track_generator_pipeline/run_test-run-guid",
        "timestamp": "2025-09-08T12:00:00.000000",
        "completed_steps": ["import_videos"],
        "context": {
            "resume_frame": 100,
            "resume_detections_count": 50,
            "resume_all_detections": [],
            "resume_crop_paths": ["path/to/crop.jpg"],
            "video_guid": "test-guid",
            "video_blob_name": "test.mp4",
            "video_folder": "test/folder"
        },
        "steps_summary": {},
        "checkpoint_version": "1.0"
    }


# Test configuration
def pytest_configure(config):
    """Configure pytest for track tests."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "checkpoint: marks tests related to checkpoint functionality"
    )
    config.addinivalue_line(
        "markers", "url_logging: marks tests related to URL logging functionality"
    )


# Custom test markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]
