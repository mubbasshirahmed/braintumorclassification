"""
utils/image_processing.py - Image Processing
============================================
Handles image preprocessing for model input
"""

import torch
from torchvision import transforms
from PIL import Image

from config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def get_transform():
    """
    Get the image transformation pipeline.
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model input.
    
    Args:
        image: PIL Image
        
    Returns:
        Tensor of shape (1, 3, 224, 224)
    """
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_transform()
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor