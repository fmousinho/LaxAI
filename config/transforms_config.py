"""
Transform configurations for training and inference.
Centralized location for all image preprocessing pipelines.
"""

import torchvision.transforms as transforms

# Image dimensions for the SiameseNet model
MODEL_INPUT_HEIGHT = 80
MODEL_INPUT_WIDTH = 40

# ImageNet normalization values (for pretrained ResNet backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transforms with data augmentation
training_transforms = transforms.Compose([
    transforms.Resize((MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)),  # Height, Width
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translation
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Inference transforms without augmentation
inference_transforms = transforms.Compose([
    transforms.Resize((MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)),  # Height, Width
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Validation transforms (same as inference, no augmentation)
validation_transforms = inference_transforms

# Transform for converting tensor back to PIL Image (for visualization)
tensor_to_pil = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in IMAGENET_STD]),  # Denormalize
    transforms.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1., 1., 1.]),  # Denormalize
    transforms.ToPILImage()
])

# Dictionary for easy access to all transforms
TRANSFORMS = {
    'training': training_transforms,
    'inference': inference_transforms,
    'validation': validation_transforms,
    'tensor_to_pil': tensor_to_pil
}

def get_transforms(mode='inference'):
    """
    Get transforms for the specified mode.
    
    Args:
        mode (str): One of 'training', 'inference', 'validation', 'tensor_to_pil'
        
    Returns:
        transforms.Compose: The requested transform pipeline
        
    Raises:
        ValueError: If mode is not recognized
    """
    if mode not in TRANSFORMS:
        raise ValueError(f"Unknown transform mode: {mode}. Available modes: {list(TRANSFORMS.keys())}")
    
    return TRANSFORMS[mode]
