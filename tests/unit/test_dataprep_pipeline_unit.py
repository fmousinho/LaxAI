"""
Unit tests for DataPrepPipeline functionality.
Tests the data preparation pipeline components without external dependencies.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock, PropertyMock
from typing import Dict, Any, Optional

from src.train.dataprep_pipeline import DataPrepPipeline


class TestDataPrepPipelineUnit:
    """Unit test suite for DataPrepP             failed_operations, successful_count = pipeline_instance._execute_parallel_operations(
            tasks, operation_func, context_info="test operations"
        )

        assert len(failed_operations) == 1
        assert successful_count == 1ed_operations, successful_count = pipeline_instance._execute_parallel_operations(
            tasks, operation_func, context_info="test operations"
        )

        assert len(failed_operations) == 1
        assert successful_count == 1 functionality."""

    @pytest.fixture
    def mock_detection_config(self):
        """Create a mock DetectionConfig for testing."""
        config = MagicMock()
        config.frames_per_video = 10
        config.gcs_bucket = "test-bucket"
        config.detection_model_path = "/path/to/model"
        config.min_resolution = (1920, 1080)
        config.max_workers = 4
        config.batch_size = 32
        return config

    @pytest.fixture
    def mock_training_config(self):
        """Create a mock TrainingConfig for testing."""
        config = MagicMock()
        config.train_ratio = 0.8
        config.num_workers = 2
        return config

    @pytest.fixture
    def mock_transform_config(self):
        """Create a mock TransformConfig for testing."""
        config = MagicMock()
        config.enable_background_removal = True
        return config

    @pytest.fixture
    def pipeline_instance(self, mock_detection_config, mock_training_config, mock_transform_config):
        """Create a DataPrepPipeline instance with mocked dependencies."""
        with patch('src.train.dataprep_pipeline.get_storage'), \
             patch('src.train.dataprep_pipeline.DetectionModel'), \
             patch('src.train.dataprep_pipeline.BackgroundMaskDetector'), \
             patch('src.train.dataprep_pipeline.GCSPaths'), \
             patch('src.train.dataprep_pipeline.training_config', mock_training_config), \
             patch('config.all_config.transform_config', mock_transform_config):

            pipeline = DataPrepPipeline(
                config=mock_detection_config,
                tenant_id="test_tenant",
                verbose=False,
                enable_grass_mask=True
            )
            return pipeline

    def test_pipeline_initialization_success(self, mock_detection_config, mock_training_config, mock_transform_config):
        """Test successful pipeline initialization."""
        with patch('src.train.dataprep_pipeline.get_storage') as mock_storage, \
             patch('src.train.dataprep_pipeline.DetectionModel') as mock_detection, \
             patch('src.train.dataprep_pipeline.BackgroundMaskDetector') as mock_bg_mask, \
             patch('src.train.dataprep_pipeline.GCSPaths') as mock_paths, \
             patch('src.train.dataprep_pipeline.training_config', mock_training_config), \
             patch('config.all_config.transform_config', mock_transform_config):

            pipeline = DataPrepPipeline(
                config=mock_detection_config,
                tenant_id="test_tenant",
                enable_grass_mask=True
            )

            assert pipeline.config == mock_detection_config
            assert pipeline.tenant_id == "test_tenant"
            assert pipeline.frames_per_video == 10
            assert pipeline.train_ratio == 0.8
            assert pipeline.enable_grass_mask is True
            assert pipeline.background_mask_detector is not None

    def test_pipeline_initialization_without_grass_mask(self, mock_detection_config, mock_training_config, mock_transform_config):
        """Test pipeline initialization without grass mask."""
        with patch('src.train.dataprep_pipeline.get_storage'), \
             patch('src.train.dataprep_pipeline.DetectionModel'), \
             patch('src.train.dataprep_pipeline.GCSPaths'), \
             patch('src.train.dataprep_pipeline.training_config', mock_training_config), \
             patch('config.all_config.transform_config', mock_transform_config):

            pipeline = DataPrepPipeline(
                config=mock_detection_config,
                tenant_id="test_tenant",
                enable_grass_mask=False
            )

            assert pipeline.enable_grass_mask is False
            assert pipeline.background_mask_detector is None

    def test_pipeline_initialization_grass_mask_none_uses_config(self, mock_detection_config, mock_training_config, mock_transform_config):
        """Test pipeline initialization with grass_mask=None uses config setting."""
        with patch('src.train.dataprep_pipeline.get_storage'), \
             patch('src.train.dataprep_pipeline.DetectionModel'), \
             patch('src.train.dataprep_pipeline.BackgroundMaskDetector'), \
             patch('src.train.dataprep_pipeline.GCSPaths'), \
             patch('src.train.dataprep_pipeline.training_config', mock_training_config), \
             patch('config.all_config.transform_config', mock_transform_config):

            pipeline = DataPrepPipeline(
                config=mock_detection_config,
                tenant_id="test_tenant",
                enable_grass_mask=None
            )

            assert pipeline.enable_grass_mask is True  # From transform_config.enable_background_removal

    def test_pipeline_initialization_detection_model_failure(self, mock_detection_config, mock_training_config, mock_transform_config):
        """Test pipeline initialization when detection model fails to load."""
        with patch('src.train.dataprep_pipeline.get_storage'), \
             patch('src.train.dataprep_pipeline.DetectionModel') as mock_detection, \
             patch('src.train.dataprep_pipeline.BackgroundMaskDetector'), \
             patch('src.train.dataprep_pipeline.GCSPaths'), \
             patch('src.train.dataprep_pipeline.training_config', mock_training_config), \
             patch('config.all_config.transform_config', mock_transform_config):

            mock_detection.side_effect = RuntimeError("Model load failed")

            with pytest.raises(RuntimeError, match="Training pipeline cannot continue"):
                DataPrepPipeline(
                    config=mock_detection_config,
                    tenant_id="test_tenant"
                )

    def test_validate_video_resolution_success(self, pipeline_instance):
        """Test video resolution validation with valid resolution."""
        # Test minimum resolution
        assert pipeline_instance._validate_video_resolution(1920, 1080) is True

        # Test higher resolution
        assert pipeline_instance._validate_video_resolution(2560, 1440) is True

    def test_validate_video_resolution_failure(self, pipeline_instance):
        """Test video resolution validation with invalid resolution."""
        # Test width too small
        assert pipeline_instance._validate_video_resolution(1280, 1080) is False

        # Test height too small
        assert pipeline_instance._validate_video_resolution(1920, 720) is False

        # Test both too small
        assert pipeline_instance._validate_video_resolution(1280, 720) is False

    def test_extract_video_metadata_success(self, pipeline_instance):
        """Test successful video metadata extraction."""
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,  # CAP_PROP_FPS
            7: 30,    # CAP_PROP_FRAME_COUNT  
            3: 1920,  # CAP_PROP_FRAME_WIDTH
            4: 1080   # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0)

        metadata = pipeline_instance._extract_video_metadata(mock_cap, "test_video.mp4")

        assert metadata['frame_count'] == 30
        assert metadata['fps'] == 30.0
        assert metadata['width'] == 1920
        assert metadata['height'] == 1080
        assert metadata['duration_seconds'] == 1.0

    def test_extract_video_metadata_failure(self, pipeline_instance):
        """Test video metadata extraction failure."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with pytest.raises(RuntimeError, match="Could not open video"):
            pipeline_instance._extract_video_metadata(mock_cap, "test_video.mp4")

    def test_import_video_success(self, pipeline_instance):
        """Test successful video import."""
        context = {'raw_video_path': 'test_video.mp4'}

        with patch.object(pipeline_instance.tenant_storage, 'move_blob') as mock_move, \
             patch.object(pipeline_instance.path_manager, 'get_path') as mock_get_path, \
             patch('src.train.dataprep_pipeline.create_video_id') as mock_create_id:

            mock_get_path.return_value = '/path/to/video/folder'
            mock_create_id.return_value = 'test_video_guid'
            mock_move.return_value = True

            result = pipeline_instance._import_video(context)

            assert result['status'] == 'completed'
            assert 'video_guid' in result
            assert result['video_guid'] == 'test_video_guid'
            assert 'video_folder' in result
            assert 'video_blob_name' in result

    def test_import_video_file_not_found(self, pipeline_instance):
        """Test video import when file is not found."""
        context = {'raw_video_path': 'nonexistent_video.mp4'}

        with patch.object(pipeline_instance.tenant_storage, 'move_blob') as mock_move:
            mock_move.return_value = False

            result = pipeline_instance._import_video(context)

            assert result['status'] == 'error'
            assert 'Failed to move video' in result['error']

    def test_detect_players_success(self, pipeline_instance):
        """Test successful player detection."""
        context = {
            'frames_data': [MagicMock()],  # Mock frame data
            'frame_ids': ['frame_001'],
            'video_guid': 'video_001',
            'video_folder': '/path/to/video/folder'
        }

        # Mock detection model
        mock_detections = MagicMock()
        mock_detections.__len__ = lambda: 5

        with patch.object(pipeline_instance.detection_model, 'detect') as mock_detect, \
             patch('supervision.Detections.from_ultralytics') as mock_from_ultralytics:

            mock_detect.return_value = MagicMock()
            mock_from_ultralytics.return_value = mock_detections

            result = pipeline_instance._detect_players(context)

            assert result['status'] == 'completed'
            assert 'all_detections' in result

    def test_detect_players_no_detections(self, pipeline_instance):
        """Test player detection when no players are detected."""
        context = {
            'frames_data': [MagicMock()],
            'frame_ids': ['frame_001'],
            'video_guid': 'video_001',
            'video_folder': '/path/to/video/folder'
        }

        mock_detections = MagicMock()
        mock_detections.__len__ = lambda: 0

        with patch.object(pipeline_instance.detection_model, 'detect') as mock_detect, \
             patch('supervision.Detections.from_ultralytics') as mock_from_ultralytics:

            mock_detect.return_value = MagicMock()
            mock_from_ultralytics.return_value = mock_detections

            result = pipeline_instance._detect_players(context)

            assert result['status'] == 'completed'
            assert 'all_detections' in result

    def test_extract_crops_success(self, pipeline_instance):
        """Test successful crop extraction."""
        import numpy as np
        import supervision as sv
        
        # Create a simple list-like mock for detections
        mock_detection = MagicMock()
        mock_detection.__getitem__.return_value = np.array([10, 20, 30, 40])  # xyxy bbox
        
        # Create a mock sv.Detections object
        mock_detections = MagicMock()
        mock_detections.__iter__.return_value = iter([mock_detection])
        mock_detections.__len__.return_value = 1
        
        # Create a mock frame as numpy array
        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        context = {
            'all_detections': [mock_detections],
            'frames_data': [mock_frame],
            'frames_guids': ['frame_001'],
            'frame_ids': ['frame_id_001'],
            'video_guid': 'video_001'
        }

        with patch.object(pipeline_instance, '_extract_crops') as mock_extract:
            mock_extract.return_value = {
                'status': 'completed',
                'crops_by_frame': [['crop_001.jpg']],
                'total_crops_extracted': 1,
                'upload_tasks': []
            }

            result = pipeline_instance._extract_crops(context)

            assert result['status'] == 'completed'
            assert 'crops_by_frame' in result
            assert 'total_crops_extracted' in result

    def test_create_training_and_validation_sets_success(self, pipeline_instance):
        """Test successful creation of training and validation sets."""
        context = {
            'crops_augmented': True,
            'frames_guids': ['frame_001'],
            'video_guid': 'video_001'
        }

        with patch('random.shuffle') as mock_shuffle, \
             patch.object(pipeline_instance.tenant_storage, 'list_blobs') as mock_list_blobs, \
             patch.object(pipeline_instance.tenant_storage, 'upload_from_string') as mock_upload, \
             patch.object(pipeline_instance.path_manager, 'get_path') as mock_get_path, \
             patch('uuid.uuid4') as mock_uuid:

            mock_list_blobs.return_value = ['player_001/', 'player_002/']
            mock_get_path.return_value = '/path/to/dataset'
            mock_uuid.return_value.hex = 'test_dataset_id'

            result = pipeline_instance._create_training_and_validation_sets(context)

            assert result['status'] == 'completed'
            assert result['datasets_created'] == True
            assert 'total_train_samples' in result
            assert 'total_val_samples' in result

    def test_run_pipeline_basic_flow(self, pipeline_instance):
        """Test basic pipeline run flow."""
        with patch.object(pipeline_instance, 'run') as mock_run:
            mock_run.return_value = {
                'status': 'completed',
                'run_guid': 'test_run_guid',
                'run_folder': 'test_run_folder',
                'video_path': 'test_video.mp4',
                'video_guid': 'test_video_guid',
                'video_folder': 'test_video_folder',
                'errors': [],
                'pipeline_summary': {'total_steps': 5, 'completed_steps': 5}
            }

            result = pipeline_instance.run('test_video.mp4')

            assert result['status'] == 'completed'
            assert 'video_guid' in result
            assert 'pipeline_summary' in result

    def test_run_pipeline_with_grass_mask(self, pipeline_instance):
        """Test pipeline run with grass mask enabled."""
        pipeline_instance.enable_grass_mask = True

        with patch.object(pipeline_instance, 'run') as mock_run:
            mock_run.return_value = {
                'status': 'completed',
                'run_guid': 'test_run_guid',
                'run_folder': 'test_run_folder',
                'video_path': 'test_video.mp4',
                'video_guid': 'test_video_guid',
                'video_folder': 'test_video_folder',
                'errors': [],
                'pipeline_summary': {'total_steps': 8, 'completed_steps': 8}
            }

            result = pipeline_instance.run('test_video.mp4')

            assert result['status'] == 'completed'

    def test_run_pipeline_step_failure(self, pipeline_instance):
        """Test pipeline run when a step fails."""
        with patch.object(pipeline_instance, 'run') as mock_run:
            mock_run.return_value = {
                'status': 'error',
                'run_guid': 'test_run_guid',
                'run_folder': 'test_run_folder',
                'video_path': 'test_video.mp4',
                'video_guid': 'unknown',
                'video_folder': 'unknown',
                'errors': ['Import failed'],
                'pipeline_summary': {'total_steps': 5, 'completed_steps': 0}
            }

            result = pipeline_instance.run('test_video.mp4')

            assert result['status'] == 'error'
            assert 'Import failed' in str(result['errors'])

    def test_execute_parallel_operations_success(self, pipeline_instance):
        """Test successful parallel operations execution."""
        tasks = [('task1', 'arg1'), ('task2', 'arg2')]
        operation_func = MagicMock(return_value='success')
        operation_func.__name__ = 'test_operation'

        failed_operations, successful_count = pipeline_instance._execute_parallel_operations(
            tasks, operation_func, context_info="test operations"
        )

        assert len(failed_operations) == 0
        assert successful_count == 2

    def test_execute_parallel_operations_with_failures(self, pipeline_instance):
        """Test parallel operations execution with some failures."""
        tasks = [('task1', 'arg1'), ('task2', 'arg2')]
        operation_func = MagicMock(side_effect=[Exception("Task 1 failed"), "Task 2 success"])
        operation_func.__name__ = 'test_operation'

        failed_operations, successful_count = pipeline_instance._execute_parallel_operations(
            tasks, operation_func, context_info="test operations"
        )

        assert len(failed_operations) == 1
        assert successful_count == 1
