"""
xai/lrp.py - LRP Implementation
==============================
Layer-wise Relevance Propagation
Using Gradient × Input method (simpler but effective)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# Simple imports
from config.settings import IMAGE_SIZE, LRP_ALPHA


def generate_lrp_explanation(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int
) -> np.ndarray:
    """
    Generate LRP-like explanation using Gradient × Input.
    
    This is a simpler approximation of LRP that works well:
    1. Compute gradients of output w.r.t. input
    2. Multiply gradients by input
    3. Shows which pixels had most influence
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image (1, 3, 224, 224)
        target_class: Class to explain (0-3)
        
    Returns:
        Relevance map as numpy array (224, 224)
    """
    
    model.eval()
    
    # Clone and enable gradients
    image_tensor = image_tensor.clone()
    image_tensor.requires_grad_(True)
    
    # Forward pass
    output = model(image_tensor)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass for target class
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot)
    
    # Get gradients
    gradients = image_tensor.grad  # (1, 3, 224, 224)
    
    # Gradient × Input
    relevance = gradients * image_tensor
    
    # Sum across channels and take absolute value
    relevance = relevance.squeeze().abs().sum(dim=0)  # (224, 224)
    
    # Convert to numpy
    relevance = relevance.detach().cpu().numpy()
    
    # Normalize to 0-1
    relevance = relevance - relevance.min()
    relevance = relevance / (relevance.max() + 1e-8)
    
    return relevance


def apply_lrp_overlay(
    original_image: Image.Image,
    relevance_map: np.ndarray,
    alpha: float = None
) -> np.ndarray:
    """
    Apply LRP relevance map as overlay on original image.
    
    Args:
        original_image: Original PIL Image
        relevance_map: LRP relevance map (224, 224)
        alpha: Transparency (0-1)
        
    Returns:
        Overlayed image as numpy array
    """
    
    if alpha is None:
        alpha = LRP_ALPHA
    
    # Resize image
    img_resized = original_image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized)
    
    # Resize relevance map if needed
    if relevance_map.shape != IMAGE_SIZE:
        relevance_map = cv2.resize(relevance_map, IMAGE_SIZE)
    
    # Apply HOT colormap (different from Grad-CAM for visual distinction)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * relevance_map),
        cv2.COLORMAP_HOT
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlayed = cv2.addWeighted(
        img_array, 1 - alpha,
        heatmap_colored, alpha,
        0
    )
    
    return overlayed