"""
Unit tests for batch detection functionality in DetectionModel.

This module tests the new batch detection feature to ensure:
1. Backward compatibility with existing single-frame detection
2. Batch processing produces equivalent results to sequential processing
3. Batch size configuration works correctly
4. Error handling for edge cases
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from supervision import Detections

from shared_libs.common.detection import DetectionModel


@pytest.fixture
def mock_rfdetr_model():
    """Create a mock RF-DETR model for testing."""
    mock_model = Mock()
    
    # Mock the predict method to return realistic Detections objects
    def mock_predict(images, threshold=0.6, **kwargs):
        # Determine if input is single image or batch
        is_batch = isinstance(images, list)
        num_images = len(images) if is_batch else 1
        
        # Create mock detections for each image
        results = []
        for i in range(num_images):
            # Create a simple Detections object with 2 boxes
            detections = Detections(
                xyxy=np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32),
                confidence=np.array([0.9, 0.8], dtype=np.float32),
                class_id=np.array([3, 3], dtype=np.int32),  # Player class
            )
            results.append(detections)
        
        # Return single or list based on input
        return results if is_batch else results[0]
    
    mock_model.predict = Mock(side_effect=mock_predict)
    mock_model.optimize_for_inference = Mock()
    
    return mock_model


@pytest.fixture
def sample_images():
    """Create sample numpy array images for testing."""
    # Create 5 dummy images (640x480 RGB)
    images = []
    for i in range(5):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        images.append(img)
    return images


class TestDetectionModelBatchSize:
    """Tests for batch size configuration and device-specific defaults."""
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_batch_size_default_cuda(self, mock_wandb, mock_rfdetr_class, mock_rfdetr_model):
        """Test that batch_size defaults to 32 for CUDA devices."""
        import torch
        
        # Mock CUDA availability
        with patch('torch.cuda.is_available', return_value=True):
            mock_rfdetr_class.return_value = mock_rfdetr_model
            
            # Mock wandb initialization
            mock_run = Mock()
            mock_artifact = Mock()
            mock_artifact.download = Mock(return_value='/tmp/test')
            mock_run.use_artifact = Mock(return_value=mock_artifact)
            mock_wandb.init.return_value = mock_run
            
            with patch('os.listdir', return_value=['model.pth']):
                with patch('os.path.exists', return_value=True):
                    with patch('os.remove'):
                        model = DetectionModel(device=torch.device('cuda'))
            
            assert model.batch_size == 32
            assert model.device.type == 'cuda'
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_batch_size_default_cpu(self, mock_wandb, mock_rfdetr_class, mock_rfdetr_model):
        """Test that batch_size defaults to 1 for CPU devices."""
        import torch
        
        # Mock CPU-only environment
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                mock_rfdetr_class.return_value = mock_rfdetr_model
                
                # Mock wandb initialization
                mock_run = Mock()
                mock_artifact = Mock()
                mock_artifact.download = Mock(return_value='/tmp/test')
                mock_run.use_artifact = Mock(return_value=mock_artifact)
                mock_wandb.init.return_value = mock_run
                
                with patch('os.listdir', return_value=['model.pth']):
                    with patch('os.path.exists', return_value=True):
                        with patch('os.remove'):
                            model = DetectionModel(device=torch.device('cpu'))
                
                assert model.batch_size == 1
                assert model.device.type == 'cpu'
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_batch_size_custom(self, mock_wandb, mock_rfdetr_class, mock_rfdetr_model):
        """Test that custom batch_size can be set."""
        import torch
        
        mock_rfdetr_class.return_value = mock_rfdetr_model
        
        # Mock wandb initialization
        mock_run = Mock()
        mock_artifact = Mock()
        mock_artifact.download = Mock(return_value='/tmp/test')
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        mock_wandb.init.return_value = mock_run
        
        with patch('os.listdir', return_value=['model.pth']):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove'):
                    model = DetectionModel(batch_size=8, device=torch.device('cuda'))
        
        assert model.batch_size == 8


class TestBatchDetectionMethod:
    """Tests for the generate_detections_batch method."""
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_batch_detection_with_multiple_images(self, mock_wandb, mock_rfdetr_class, 
                                                   mock_rfdetr_model, sample_images):
        """Test batch detection with multiple images."""
        import torch
        
        mock_rfdetr_class.return_value = mock_rfdetr_model
        
        # Mock wandb initialization
        mock_run = Mock()
        mock_artifact = Mock()
        mock_artifact.download = Mock(return_value='/tmp/test')
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        mock_wandb.init.return_value = mock_run
        
        with patch('os.listdir', return_value=['model.pth']):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove'):
                    model = DetectionModel(batch_size=2, device=torch.device('cuda'))
        
        # Test batch detection
        results = model.generate_detections_batch(sample_images)
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) == 5  # Should get results for all 5 images
        
        for detection in results:
            assert isinstance(detection, Detections)
            assert len(detection.xyxy) == 2  # Each should have 2 boxes
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_batch_detection_with_single_image(self, mock_wandb, mock_rfdetr_class, 
                                                mock_rfdetr_model, sample_images):
        """Test batch detection with a single image (edge case)."""
        import torch
        
        mock_rfdetr_class.return_value = mock_rfdetr_model
        
        # Mock wandb initialization
        mock_run = Mock()
        mock_artifact = Mock()
        mock_artifact.download = Mock(return_value='/tmp/test')
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        mock_wandb.init.return_value = mock_run
        
        with patch('os.listdir', return_value=['model.pth']):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove'):
                    model = DetectionModel(batch_size=4, device=torch.device('cuda'))
        
        # Test with single image
        results = model.generate_detections_batch([sample_images[0]])
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], Detections)
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_batch_detection_empty_list(self, mock_wandb, mock_rfdetr_class, mock_rfdetr_model):
        """Test batch detection with empty list raises ValueError."""
        import torch
        
        mock_rfdetr_class.return_value = mock_rfdetr_model
        
        # Mock wandb initialization
        mock_run = Mock()
        mock_artifact = Mock()
        mock_artifact.download = Mock(return_value='/tmp/test')
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        mock_wandb.init.return_value = mock_run
        
        with patch('os.listdir', return_value=['model.pth']):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove'):
                    model = DetectionModel(device=torch.device('cuda'))
        
        with pytest.raises(ValueError, match="Images list cannot be empty"):
            model.generate_detections_batch([])
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_batch_detection_batching_logic(self, mock_wandb, mock_rfdetr_class, 
                                            mock_rfdetr_model, sample_images):
        """Test that images are properly batched according to batch_size."""
        import torch
        
        mock_rfdetr_class.return_value = mock_rfdetr_model
        
        # Mock wandb initialization
        mock_run = Mock()
        mock_artifact = Mock()
        mock_artifact.download = Mock(return_value='/tmp/test')
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        mock_wandb.init.return_value = mock_run
        
        with patch('os.listdir', return_value=['model.pth']):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove'):
                    # Set batch size to 2, so 5 images should result in 3 calls
                    # (batch of 2, batch of 2, batch of 1)
                    model = DetectionModel(batch_size=2, device=torch.device('cuda'))
        
        # Call batch detection
        results = model.generate_detections_batch(sample_images)
        
        # Verify the predict method was called 3 times (ceiling of 5/2)
        assert mock_rfdetr_model.predict.call_count == 3
        assert len(results) == 5


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_single_frame_detection_still_works(self, mock_wandb, mock_rfdetr_class, 
                                                mock_rfdetr_model, sample_images):
        """Test that existing generate_detections method still works."""
        import torch
        
        mock_rfdetr_class.return_value = mock_rfdetr_model
        
        # Mock wandb initialization
        mock_run = Mock()
        mock_artifact = Mock()
        mock_artifact.download = Mock(return_value='/tmp/test')
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        mock_wandb.init.return_value = mock_run
        
        with patch('os.listdir', return_value=['model.pth']):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove'):
                    model = DetectionModel(device=torch.device('cuda'))
        
        # Test single frame detection (original method)
        result = model.generate_detections(sample_images[0])
        
        assert isinstance(result, Detections)
        assert len(result.xyxy) == 2
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_model_initialization_without_batch_size(self, mock_wandb, mock_rfdetr_class, 
                                                     mock_rfdetr_model):
        """Test that DetectionModel can still be initialized without batch_size parameter."""
        import torch
        
        with patch('torch.cuda.is_available', return_value=True):
            mock_rfdetr_class.return_value = mock_rfdetr_model
            
            # Mock wandb initialization
            mock_run = Mock()
            mock_artifact = Mock()
            mock_artifact.download = Mock(return_value='/tmp/test')
            mock_run.use_artifact = Mock(return_value=mock_artifact)
            mock_wandb.init.return_value = mock_run
            
            with patch('os.listdir', return_value=['model.pth']):
                with patch('os.path.exists', return_value=True):
                    with patch('os.remove'):
                        # Old initialization style (no batch_size parameter)
                        model = DetectionModel()
            
            # Should still work and use default batch_size based on device
            assert hasattr(model, 'batch_size')
            assert model.batch_size == 32  # Default for CUDA


class TestBatchVsSequentialEquivalence:
    """Tests to verify batch processing produces equivalent results to sequential processing."""
    
    @patch('shared_libs.common.detection.RFDETRBase')
    @patch('shared_libs.common.detection.wandb')
    def test_batch_equals_sequential(self, mock_wandb, mock_rfdetr_class, 
                                     mock_rfdetr_model, sample_images):
        """Test that batch detection produces same results as sequential detection."""
        import torch
        
        mock_rfdetr_class.return_value = mock_rfdetr_model
        
        # Mock wandb initialization
        mock_run = Mock()
        mock_artifact = Mock()
        mock_artifact.download = Mock(return_value='/tmp/test')
        mock_run.use_artifact = Mock(return_value=mock_artifact)
        mock_wandb.init.return_value = mock_run
        
        with patch('os.listdir', return_value=['model.pth']):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove'):
                    model = DetectionModel(batch_size=2, device=torch.device('cuda'))
        
        # Get results from batch processing
        batch_results = model.generate_detections_batch(sample_images[:3])
        
        # Get results from sequential processing
        sequential_results = []
        for img in sample_images[:3]:
            result = model.generate_detections(img)
            sequential_results.append(result)
        
        # Verify same number of results
        assert len(batch_results) == len(sequential_results)
        
        # Verify each result has same structure
        for batch_det, seq_det in zip(batch_results, sequential_results):
            assert len(batch_det.xyxy) == len(seq_det.xyxy)
            assert np.array_equal(batch_det.xyxy, seq_det.xyxy)
            if batch_det.confidence is not None and seq_det.confidence is not None:
                assert np.array_equal(batch_det.confidence, seq_det.confidence)
            if batch_det.class_id is not None and seq_det.class_id is not None:
                assert np.array_equal(batch_det.class_id, seq_det.class_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
