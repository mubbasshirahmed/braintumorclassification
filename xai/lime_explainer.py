"""
xai/lime_explainer.py - LIME Implementation
==========================================
Local Interpretable Model-agnostic Explanations
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Simple imports - matches gradcam.py and lrp.py
from config.settings import IMAGE_SIZE, LIME_NUM_SAMPLES, LIME_NUM_FEATURES
from utils.image_processing import preprocess_image


def generate_lime_explanation(
    model: nn.Module,
    image: Image.Image,
    target_class: int,
    num_samples: int = None,
    num_features: int = None
) -> np.ndarray:
    """
    Generate LIME explanation.
    
    LIME works by:
    1. Dividing image into superpixels
    2. Creating variations by hiding superpixels
    3. Observing how predictions change
    4. Identifying important superpixels
    
    Args:
        model: Trained PyTorch model
        image: Original PIL Image
        target_class: Class to explain (0-3)
        num_samples: Number of perturbed images (default: 1000)
        num_features: Number of superpixels to highlight (default: 10)
        
    Returns:
        Image with superpixel boundaries
    """
    
    num_samples = num_samples or LIME_NUM_SAMPLES
    num_features = num_features or LIME_NUM_FEATURES
    
    # Resize image
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized)
    
    # Prediction function for LIME
    def predict_fn(images: np.ndarray) -> np.ndarray:
        """Predict probabilities for batch of images."""
        
        batch_tensors = []
        
        for img in images:
            pil_img = Image.fromarray(img.astype('uint8'))
            tensor = preprocess_image(pil_img)
            batch_tensors.append(tensor)
        
        batch = torch.cat(batch_tensors, dim=0)
        
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples
    )
    
    # Get mask for target class
    temp, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    
    # Draw boundaries
    img_with_boundaries = mark_boundaries(temp / 255.0, mask)
    
    return img_with_boundaries