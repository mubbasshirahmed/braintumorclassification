"""
models/predictor.py - Prediction Logic
======================================
Handles making predictions with trained models
"""

import torch
import torch.nn as nn
import numpy as np

from config import CLASS_NAMES


def predict(model: nn.Module, image_tensor: torch.Tensor) -> dict:
    """
    Make a prediction on an image tensor.
    
    Args:
        model: Loaded PyTorch model
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
        
    Returns:
        Dictionary with:
        - 'class': int (0-3)
        - 'label': str ("Glioma", etc.)
        - 'confidence': float (0-1)
        - 'probabilities': numpy array
    """
    
    # Ensure model is in eval mode
    model.eval()
    
    # No gradient computation needed
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Get confidence
        confidence = probabilities[0][predicted_class].item()
    
    return {
        'class': predicted_class,
        'label': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities[0].numpy()
    }