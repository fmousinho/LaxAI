#!/usr/bin/env python3
"""
Example demonstrating how to use background removal transforms in a training pipeline.

This example shows how to integrate background removal into the existing training
workflow, comparing results with and without background removal.
"""

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.background_mask import BackgroundMaskDetector, create_frame_generator_from_images
from config.transforms import get_transforms, create_transforms_with_background_removal
from config.all_config import BackgroundMaskConfig


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing background removal transforms.
    """
    
    def __init__(self, num_samples=100, image_size=(200, 200), transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        
        # Create synthetic data
        self.data = []
        self.labels = []
        
        for i in range(num_samples):
            # Create image with green background
            image = np.zeros((*image_size, 3), dtype=np.uint8)
            image[:, :] = [0, 255, 0]  # Green background
            
            # Add random objects with different colors
            if i % 3 == 0:
                # Red square
                x, y = np.random.randint(20, image_size[0]-60), np.random.randint(20, image_size[1]-60)
                image[x:x+40, y:y+40] = [255, 0, 0]
                label = 0
            elif i % 3 == 1:
                # Blue circle
                center_x, center_y = np.random.randint(30, image_size[0]-30), np.random.randint(30, image_size[1]-30)
                radius = 20
                y_coords, x_coords = np.ogrid[:image_size[0], :image_size[1]]
                mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                image[mask] = [0, 0, 255]
                label = 1
            else:
                # Yellow triangle (approximate)
                x, y = np.random.randint(20, image_size[0]-60), np.random.randint(20, image_size[1]-60)
                for j in range(40):
                    image[x+j, y+j//2:y+40-j//2] = [255, 255, 0]
                label = 2
            
            self.data.append(image)
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image for transforms
        pil_image = Image.fromarray(image)
        
        if self.transform:
            pil_image = self.transform(pil_image)
        
        return pil_image, torch.tensor(label, dtype=torch.long)


def analyze_dataset_statistics(dataloader, name):
    """Analyze dataset statistics."""
    print(f"\n{name} Dataset Statistics:")
    print("-" * 30)
    
    total_samples = 0
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        batch_size = images.size(0)
        total_samples += batch_size
        
        # Calculate statistics
        pixel_sum += images.sum(dim=[0, 2, 3])
        pixel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        
        if batch_idx == 0:
            print(f"  Batch shape: {images.shape}")
            print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Calculate mean and std
    total_pixels = total_samples * images.size(2) * images.size(3)
    mean = pixel_sum / total_pixels
    std = torch.sqrt(pixel_squared_sum / total_pixels - mean ** 2)
    
    print(f"  Total samples: {total_samples}")
    print(f"  Channel means: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"  Channel stds: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")


def main():
    print("Background Removal Training Pipeline Example")
    print("=" * 60)
    
    # Create training background detector
    print("\n1. Creating background detector...")
    
    # Create sample images for background detection
    bg_sample_images = []
    for i in range(10):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :] = [0, 255, 0]  # Green background
        
        # Add some random objects
        if i % 2 == 0:
            x, y = np.random.randint(20, 160), np.random.randint(20, 160)
            img[x:x+40, y:y+40] = [255, 0, 0]  # Red square
        
        bg_sample_images.append(img)
    
    frame_generator = create_frame_generator_from_images(bg_sample_images)
    
    bg_config = BackgroundMaskConfig(
        sample_frames=5,
        std_dev_multiplier=1.0,
        replacement_color=(255, 255, 255),  # White replacement
        top_crop_ratio=0.2,
        bottom_crop_ratio=0.1,
        verbose=True
    )
    
    background_detector = BackgroundMaskDetector(
        frame_generator=frame_generator,
        config=bg_config
    )
    
    print("✓ Background detector created successfully")
    
    # Create datasets
    print("\n2. Creating datasets...")
    
    # Standard training dataset
    standard_transforms = get_transforms('training')
    standard_dataset = SyntheticDataset(
        num_samples=50,
        transform=standard_transforms
    )
    
    # Background removal training dataset
    bg_transforms = get_transforms('training', background_detector=background_detector)
    bg_dataset = SyntheticDataset(
        num_samples=50,
        transform=bg_transforms
    )
    
    print(f"✓ Created datasets with {len(standard_dataset)} samples each")
    
    # Create data loaders
    print("\n3. Creating data loaders...")
    
    standard_dataloader = DataLoader(standard_dataset, batch_size=8, shuffle=True)
    bg_dataloader = DataLoader(bg_dataset, batch_size=8, shuffle=True)
    
    print("✓ Data loaders created")
    
    # Analyze dataset statistics
    print("\n4. Analyzing dataset statistics...")
    
    analyze_dataset_statistics(standard_dataloader, "Standard")
    analyze_dataset_statistics(bg_dataloader, "Background Removal")
    
    # Test inference
    print("\n5. Testing inference transforms...")
    
    # Create test image
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)
    test_image[:, :] = [0, 255, 0]  # Green background
    test_image[50:150, 50:150] = [255, 0, 0]  # Red square
    
    test_pil = Image.fromarray(test_image)
    
    # Standard inference
    standard_inference = get_transforms('inference')
    standard_result = standard_inference(test_pil)
    
    # Background removal inference
    bg_inference = get_transforms('inference', background_detector=background_detector)
    bg_result = bg_inference(test_pil)
    
    print(f"Standard inference result shape: {standard_result.shape}")
    print(f"Standard inference range: [{standard_result.min():.3f}, {standard_result.max():.3f}]")
    print(f"Background removal inference result shape: {bg_result.shape}")
    print(f"Background removal inference range: [{bg_result.min():.3f}, {bg_result.max():.3f}]")
    
    # Test OpenCV-style input
    print("\n6. Testing OpenCV-style input...")
    
    # Simulate OpenCV BGR input
    bgr_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR) if 'cv2' in globals() else test_image
    
    opencv_standard = get_transforms('opencv_safe')
    opencv_bg = get_transforms('opencv_safe', background_detector=background_detector)
    
    opencv_standard_result = opencv_standard(test_image)  # Use RGB since cv2 might not be available
    opencv_bg_result = opencv_bg(test_image)
    
    print(f"OpenCV standard result shape: {opencv_standard_result.shape}")
    print(f"OpenCV background removal result shape: {opencv_bg_result.shape}")
    
    # Performance comparison
    print("\n7. Performance comparison...")
    
    import time
    
    # Time standard transforms
    start_time = time.time()
    for i in range(10):
        _ = standard_transforms(test_pil)
    standard_time = time.time() - start_time
    
    # Time background removal transforms
    start_time = time.time()
    for i in range(10):
        _ = bg_transforms(test_pil)
    bg_time = time.time() - start_time
    
    print(f"Standard transforms: {standard_time:.4f}s for 10 images")
    print(f"Background removal transforms: {bg_time:.4f}s for 10 images")
    print(f"Background removal overhead: {(bg_time - standard_time) / standard_time * 100:.1f}%")
    
    print("\n✅ All background removal training pipeline tests completed!")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Background detector integration successful")
    print("✓ Training and inference transforms working")
    print("✓ OpenCV-safe transforms working")
    print("✓ Dataset statistics analysis complete")
    print("✓ Performance benchmarking complete")
    print("\nThe background removal transforms are ready for use in your training pipeline!")


if __name__ == "__main__":
    main()
