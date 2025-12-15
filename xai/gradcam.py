"""
xai/gradcam.py - Grad-CAM Implementation
========================================
Gradient-weighted Class Activation Mapping
Manual implementation - NO Captum!
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# Simple imports
from config.settings import IMAGE_SIZE, GRADCAM_ALPHA


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Shows which regions of the image are important for the prediction.
    
    How it works:
    1. Forward pass to get prediction
    2. Backward pass to get gradients for target class
    3. Weight feature maps by gradients
    4. Create heatmap of important regions
    """
    
    def __init__(self, model: nn.Module, model_name: str):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained PyTorch model
            model_name: "EfficientNetB0" or "VGG16"
        """
        self.model = model
        self.model_name = model_name
        
        # Storage for hooks
        self.activations = None
        self.gradients = None
        
        # Get target layer
        self.target_layer = self._get_target_layer()
        
        # Register hooks
        self._register_hooks()
    
    def _get_target_layer(self) -> nn.Module:
        """Get the last convolutional layer."""
        
        if "efficientnet" in self.model_name.lower():
            return self.model.features[-1]
        elif "vgg" in self.model_name.lower():
            # VGG: features[-1] is MaxPool, features[-3] is last Conv2d
            return self.model.features[-3]
        else:
            return self.model.features[-1]
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image_tensor: Preprocessed image (1, 3, 224, 224)
            target_class: Class to explain (0-3)
            
        Returns:
            Heatmap as numpy array (values 0-1)
        """
        
        self.model.eval()
        
        # Enable gradients for input
        image_tensor = image_tensor.clone()
        image_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get activations and gradients
        activations = self.activations  # (1, C, H, W)
        gradients = self.gradients      # (1, C, H, W)
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU
        cam = torch.relu(cam)
        
        # Convert to numpy
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam


def apply_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = None
) -> np.ndarray:
    """
    Apply Grad-CAM heatmap as overlay on original image.
    
    Args:
        original_image: Original PIL Image
        heatmap: Grad-CAM heatmap
        alpha: Transparency (0-1)
        
    Returns:
        Overlayed image as numpy array
    """
    
    if alpha is None:
        alpha = GRADCAM_ALPHA
    
    # Resize image to 224x224
    img_resized = original_image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized)
    
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
    
    # Apply colormap (JET: blue=cold, red=hot)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlayed = cv2.addWeighted(
        img_array, 1 - alpha,
        heatmap_colored, alpha,
        0
    )
    
    return overlayed