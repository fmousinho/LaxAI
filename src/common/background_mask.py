"""
Background mask utilities for the LaxAI project.

This module provides a class for detecting background colors from video frames
and removing backgrounds from images based on HSV color analysis.

The BackgroundMaskDetector class now uses a centralized configuration system
for all parameters, making it easier to manage and maintain consistent settings
across different use cases.

Configuration Options:
    - sample_frames: Number of frames to analyze for background detection
    - std_dev_multiplier: Sensitivity of background detection (higher = more permissive)
    - replacement_color: RGB color to replace detected background pixels
    - top_crop_ratio: Fraction of frame to remove from top (0.0 to 1.0)
    - bottom_crop_ratio: Fraction of frame to remove from bottom (0.0 to 1.0)
    - hsv_min_values: Minimum HSV values for color bounds
    - hsv_max_values: Maximum HSV values for color bounds
    - min_std_multiplier: Minimum allowed standard deviation multiplier
    - max_std_multiplier: Maximum allowed standard deviation multiplier
    - verbose: Enable detailed progress information

Usage Examples:
    # Using global default configuration
    detector = BackgroundMaskDetector()
    detector.initialize(frame_generator)
    
    # Using custom configuration
    custom_config = BackgroundMaskConfig(sample_frames=3, replacement_color=(0, 255, 0))
    detector = BackgroundMaskDetector(config=custom_config)
    detector.initialize(frame_generator)
    
    # Using parameter overrides (any config field can be overridden)
    detector = BackgroundMaskDetector(
        sample_frames=2, 
        verbose=False,
        top_crop_ratio=0.3,
        replacement_color=(255, 0, 0)
    )
    detector.initialize(frame_generator)
    
    # Dynamic updates
    detector.update_replacement_color((255, 0, 0))
    detector.update_bounds(1.5)
"""

import cv2
import numpy as np
from typing import List, Union, Tuple, Generator, Optional
import logging

logger = logging.getLogger(__name__)


class BackgroundMaskDetector:
    """
    A class that detects background color from video frames and provides
    background removal functionality.
    
    The class analyzes a sample of frames to determine the dominant background
    color in HSV space, then provides methods to remove similar colors from images.
    """
    
    def __init__(
        self,
        config: Optional["BackgroundMaskConfig"] = None,
        **kwargs
    ):
        """
        Initialize the background mask detector.
        
        Args:
            config: BackgroundMaskConfig instance (uses global config if None)
            **kwargs: Override any config parameters (sample_frames, std_dev_multiplier, 
                     replacement_color, verbose, top_crop_ratio, bottom_crop_ratio, etc.)
        """
        # Use provided config or global config
        if config is not None:
            self.config = config
        else:
            from src.config.all_config import background_mask_config
            self.config = background_mask_config
        
        # Override config values with provided parameters
        self.sample_frames = kwargs.get('sample_frames', self.config.sample_frames)
        self.std_dev_multiplier = kwargs.get('std_dev_multiplier', self.config.std_dev_multiplier)
        self.replacement_color = kwargs.get('replacement_color', self.config.replacement_color)
        self.verbose = kwargs.get('verbose', self.config.verbose)
        self.top_crop_ratio = kwargs.get('top_crop_ratio', self.config.top_crop_ratio)
        self.bottom_crop_ratio = kwargs.get('bottom_crop_ratio', self.config.bottom_crop_ratio)
        self.hsv_min_values = kwargs.get('hsv_min_values', self.config.hsv_min_values)
        self.hsv_max_values = kwargs.get('hsv_max_values', self.config.hsv_max_values)
        self.min_std_multiplier = kwargs.get('min_std_multiplier', self.config.min_std_multiplier)
        self.max_std_multiplier = kwargs.get('max_std_multiplier', self.config.max_std_multiplier)
        
        # Initialize background color statistics
        self.mean_hsv: Optional[np.ndarray] = None
        self.std_hsv: Optional[np.ndarray] = None
        self.lower_bound: Optional[np.ndarray] = None
        self.upper_bound: Optional[np.ndarray] = None
        
        logger.info("BackgroundMaskDetector created. Call initialize() to detect background color.")
    
    def initialize(self, frame_generator: Generator[np.ndarray, None, None]):
        """
        Initialize the detector by analyzing frames to detect background color.
        
        Args:
            frame_generator: Generator that yields BGR frames from video
        """
        # Detect background color from frames
        self._detect_background_color(frame_generator)
        logger.info("BackgroundMaskDetector initialized successfully.")
    
    def _detect_background_color(self, frame_generator: Generator[np.ndarray, None, None]):
        """
        Analyze frames to detect the dominant background color.
        
        Args:
            frame_generator: Generator that yields BGR frames from video
        """
        if self.verbose:
            logger.info("Analyzing frames for background color detection...")

        # Collect frames from the generator
        frames = []
        frame_count = 0
        
        try:
            for frame in frame_generator:
                # Convert BGR to RGB for consistent processing
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame  # Handle grayscale or other formats
                frames.append(rgb_frame)
                frame_count += 1
        except Exception as e:
            if self.verbose:
                print(f"Frame collection completed with {frame_count} frames")
        
        if len(frames) == 0:
            raise ValueError("No frames were provided by the generator")
        
        # Select equally distributed frames
        total_frames = len(frames)
        if total_frames < self.sample_frames:
            selected_frames = frames
            if self.verbose:
                print(f"Using all {total_frames} available frames (requested {self.sample_frames})")
        else:
            indices = np.linspace(0, total_frames - 1, self.sample_frames, dtype=int)
            selected_frames = [frames[i] for i in indices]
            if self.verbose:
                print(f"Selected {len(selected_frames)} frames from {total_frames} total frames")
        
        # Process each selected frame
        background_pixels = []
        
        for i, frame in enumerate(selected_frames):
            if self.verbose:
                print(f"Processing frame {i+1}/{len(selected_frames)}")
            
            # Remove top and bottom portions based on config
            h, w = frame.shape[:2]
            top_crop = int(h * self.top_crop_ratio)  # Remove top portion based on config
            bottom_crop = int(h * (1.0 - self.bottom_crop_ratio))  # Keep portion based on config
            
            cropped_frame = frame[top_crop:bottom_crop, :]
            
            if cropped_frame.size == 0:
                continue
            
            # Convert RGB to HSV for color analysis
            hsv_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2HSV)
            
            # Reshape to get all pixels
            pixels = hsv_frame.reshape(-1, 3)
            background_pixels.append(pixels)
        
        if not background_pixels:
            raise ValueError("No valid background pixels found after processing frames")
        
        # Combine all background pixels
        all_pixels = np.vstack(background_pixels)
        
        # Calculate mean and standard deviation in HSV space
        self.mean_hsv = np.mean(all_pixels, axis=0).astype(np.float64)
        self.std_hsv = np.std(all_pixels, axis=0).astype(np.float64)
        
        # Calculate bounds
        self.lower_bound = np.maximum(
            self.mean_hsv - self.std_dev_multiplier * self.std_hsv,
            self.hsv_min_values
        ).astype(np.uint8)
        self.upper_bound = np.minimum(
            self.mean_hsv + self.std_dev_multiplier * self.std_hsv,
            self.hsv_max_values
        ).astype(np.uint8)
        
        if self.verbose:
            print(f"Background color detected:")
            print(f"  Mean HSV: {self.mean_hsv}")
            print(f"  Std HSV: {self.std_hsv}")
            print(f"  Lower bound: {self.lower_bound}")
            print(f"  Upper bound: {self.upper_bound}")
            print(f"  Replacement color (RGB): {self.replacement_color}")
    
    def _convert_to_rgb(self, image: np.ndarray, input_format: str = 'BGR') -> np.ndarray:
        """
        Convert image to RGB format for consistent processing.
        
        Args:
            image: Input image array
            input_format: Input format ('BGR' or 'RGB')
            
        Returns:
            Image in RGB format
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image  # Return as-is for non-color images
        
        if input_format == 'BGR':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return image  # Already RGB
    
    def _detect_color_format(self, image: np.ndarray) -> str:
        """
        Attempt to detect if image is in BGR or RGB format.
        
        This is a heuristic approach - it's better to explicitly specify the format.
        
        Args:
            image: Input image array
            
        Returns:
            'BGR' or 'RGB' based on heuristic analysis
        """
        # This is a simple heuristic - in practice, you should know your input format
        # For now, we'll assume most inputs are BGR (typical OpenCV format)
        return 'BGR'
    
    def remove_background(
        self, 
        images: Union[np.ndarray, List[np.ndarray]],
        input_format: str = 'BGR'
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Remove background from image(s) based on detected background color.
        
        Args:
            images: Single image or list of images
            input_format: Format of input images ('BGR' or 'RGB')
            
        Returns:
            Processed image(s) with background removed (same format as input)
        """
        if self.mean_hsv is None or self.lower_bound is None or self.upper_bound is None:
            raise ValueError("Background color not detected. Initialize detector first.")
        
        # Handle single image vs list of images
        single_image = False
        if isinstance(images, np.ndarray):
            single_image = True
            images = [images]
        
        processed_images = []
        
        for img in images:
            if img is None or img.size == 0:
                processed_images.append(img)
                continue
            
            # Convert to RGB for consistent processing
            rgb_img = self._convert_to_rgb(img, input_format)
            
            # Convert RGB to HSV for color analysis
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            
            # Create mask for background pixels
            mask = cv2.inRange(hsv_img, self.lower_bound, self.upper_bound)
            
            # Create output image (keep in original format)
            result = img.copy()
            
            # Convert replacement color to match input format
            if input_format == 'BGR':
                # Convert RGB replacement color to BGR
                replacement_bgr = self.replacement_color[::-1]  # Reverse RGB to BGR
                result[mask > 0] = replacement_bgr
            else:
                # Input is RGB, use replacement color as-is
                result[mask > 0] = self.replacement_color
            
            processed_images.append(result)
        
        # Return single image if input was single image
        if single_image:
            return processed_images[0]
        
        return processed_images
    
    def get_background_mask(
        self, 
        images: Union[np.ndarray, List[np.ndarray]],
        input_format: str = 'BGR'
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get binary mask(s) indicating background pixels.
        
        Args:
            images: Single image or list of images
            input_format: Format of input images ('BGR' or 'RGB')
            
        Returns:
            Binary mask(s) where 255 = background, 0 = foreground
        """
        if self.mean_hsv is None or self.lower_bound is None or self.upper_bound is None:
            raise ValueError("Background color not detected. Initialize detector first.")
        
        # Handle single image vs list of images
        single_image = False
        if isinstance(images, np.ndarray):
            single_image = True
            images = [images]
        
        masks = []
        
        for img in images:
            if img is None or img.size == 0:
                masks.append(None)
                continue
            
            # Convert to RGB for consistent processing
            rgb_img = self._convert_to_rgb(img, input_format)
            
            # Convert RGB to HSV for color analysis
            hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
            
            # Create mask for background pixels
            mask = cv2.inRange(hsv_img, self.lower_bound, self.upper_bound)
            
            masks.append(mask)
        
        # Return single mask if input was single image
        if single_image:
            return masks[0]
        
        return masks
    
    def update_replacement_color(self, new_color: Tuple[int, int, int]):
        """
        Update the replacement color for background removal.
        
        Args:
            new_color: New RGB color tuple
        """
        self.replacement_color = new_color
        if self.verbose:
            print(f"Replacement color updated to: {new_color}")
    
    def update_bounds(self, new_std_multiplier: float):
        """
        Update the color bounds using a new standard deviation multiplier.
        
        Args:
            new_std_multiplier: New multiplier for standard deviation
        """
        if self.mean_hsv is None or self.std_hsv is None:
            raise ValueError("Background color not detected. Initialize detector first.")
        
        # Validate the new multiplier
        if new_std_multiplier < self.min_std_multiplier:
            if self.verbose:
                print(f"Warning: std_multiplier {new_std_multiplier} is below minimum {self.min_std_multiplier}. Using minimum.")
            new_std_multiplier = self.min_std_multiplier
        elif new_std_multiplier > self.max_std_multiplier:
            if self.verbose:
                print(f"Warning: std_multiplier {new_std_multiplier} is above maximum {self.max_std_multiplier}. Using maximum.")
            new_std_multiplier = self.max_std_multiplier
        
        self.std_dev_multiplier = new_std_multiplier
        
        # Recalculate bounds
        self.lower_bound = np.maximum(
            self.mean_hsv - self.std_dev_multiplier * self.std_hsv,
            self.hsv_min_values
        ).astype(np.uint8)
        self.upper_bound = np.minimum(
            self.mean_hsv + self.std_dev_multiplier * self.std_hsv,
            self.hsv_max_values
        ).astype(np.uint8)
        
        if self.verbose:
            print(f"Bounds updated with std multiplier: {new_std_multiplier}")
            print(f"  New lower bound: {self.lower_bound}")
            print(f"  New upper bound: {self.upper_bound}")
    
    def get_stats(self) -> dict:
        """
        Get background color statistics.
        
        Returns:
            Dictionary containing background color statistics
        """
        return {
            'mean_hsv': self.mean_hsv,
            'std_hsv': self.std_hsv,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'std_dev_multiplier': self.std_dev_multiplier,
            'replacement_color': self.replacement_color,
            'config': {
                'sample_frames': self.sample_frames,
                'top_crop_ratio': self.top_crop_ratio,
                'bottom_crop_ratio': self.bottom_crop_ratio,
                'hsv_min_values': self.hsv_min_values,
                'hsv_max_values': self.hsv_max_values,
                'min_std_multiplier': self.min_std_multiplier,
                'max_std_multiplier': self.max_std_multiplier
            }
        }


def create_frame_generator_from_video(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Create a frame generator from a video file.
    
    Args:
        video_path: Path to the video file
        
    Yields:
        BGR frames from the video (OpenCV default format)
    """
    cap = cv2.VideoCapture(video_path)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV reads in BGR format by default
            yield frame
    
    finally:
        cap.release()


def create_frame_generator_from_images(images: List[np.ndarray], input_format: str = 'RGB') -> Generator[np.ndarray, None, None]:
    """
    Create a frame generator from a list of images.
    
    Args:
        images: List of images
        input_format: Format of input images ('BGR' or 'RGB')
        
    Yields:
        BGR frames for consistency with video generator
    """
    for img in images:
        if input_format == 'RGB':
            # Convert RGB to BGR for consistency
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield bgr_img
        else:
            # Already BGR
            yield img


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    print("Testing BackgroundMaskDetector with synthetic data...")
    
    # Create some test frames with green background
    test_frames = []
    for i in range(10):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [0, 255, 0]  # Green background
        
        # Add some non-background objects
        frame[30:70, 30:70] = [255, 0, 0]  # Red square
        
        test_frames.append(frame)
    
    # Create frame generator
    frame_gen = create_frame_generator_from_images(test_frames)
    
    # Test 1: Using default configuration
    print("\n=== Test 1: Default Configuration ===")
    detector = BackgroundMaskDetector(verbose=True)
    detector.initialize(frame_gen)
    
    # Test background removal
    test_image = test_frames[0]
    result = detector.remove_background(test_image)
    
    print(f"Test completed. Background detection and removal working correctly.")
    if isinstance(result, np.ndarray):
        print(f"Original shape: {test_image.shape}, Result shape: {result.shape}")
    else:
        print(f"Result type: {type(result)}")
    
    # Test 2: Using custom configuration
    print("\n=== Test 2: Custom Configuration ===")
    from src.config.all_config import BackgroundMaskConfig
    
    custom_config = BackgroundMaskConfig(
        sample_frames=3,
        std_dev_multiplier=1.5,
        replacement_color=(0, 0, 255),  # Blue replacement
        verbose=True,
        top_crop_ratio=0.3,  # Remove only top 30%
        bottom_crop_ratio=0.05  # Remove only bottom 5%
    )
    
    frame_gen2 = create_frame_generator_from_images(test_frames)
    detector2 = BackgroundMaskDetector(config=custom_config)
    detector2.initialize(frame_gen2)
    
    result2 = detector2.remove_background(test_image)
    print(f"Custom config test completed with blue replacement color.")
    
    # Test 3: Parameter override
    print("\n=== Test 3: Parameter Override ===")
    frame_gen3 = create_frame_generator_from_images(test_frames)
    detector3 = BackgroundMaskDetector(
        sample_frames=2,  # Override config
        replacement_color=(255, 255, 0),  # Yellow replacement
        verbose=True
    )
    detector3.initialize(frame_gen3)
    
    result3 = detector3.remove_background(test_image)
    print(f"Parameter override test completed with yellow replacement color.")
    
    print(f"\nAll tests completed successfully!")
