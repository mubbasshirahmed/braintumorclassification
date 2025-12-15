"""
config/settings.py - All Configuration Settings
================================================
"""

# Class names for tumor types
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Model paths
MODEL_PATHS = {
    "EfficientNetB0": "models/efficientnet_b0_trained.pth",
    "VGG16": "models/vgg16_trained.pth"
}

# Image settings
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# XAI settings
LIME_NUM_SAMPLES = 1000
LIME_NUM_FEATURES = 10
GRADCAM_ALPHA = 0.4
LRP_ALPHA = 0.4

# Supported formats
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']

# Model accuracy (from your dissertation)
MODEL_ACCURACY = {
    "EfficientNetB0": "95.2%",
    "VGG16": "93.8%"
}

XAI_ACCURACY = {
    "Grad-CAM": "92%",
    "LIME": "85%",
    "LRP": "88%"
}