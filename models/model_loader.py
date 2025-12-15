
import torch
import torch.nn as nn
import streamlit as st
from torchvision import models
from typing import Optional

# Import configuration
from config import MODEL_PATHS

# Import downloader
from utils.model_downloader import download_model_from_gdrive


@st.cache_resource
def load_model(model_name: str) -> Optional[nn.Module]:
    """
    Load trained model with automatic Google Drive download.
    
    Args:
        model_name: "EfficientNetB0" or "VGG16"
        
    Returns:
        Loaded PyTorch model or None if loading fails
    """
    
    try:
        # Step 1: Download model from Google Drive if needed
        model_path = download_model_from_gdrive(model_name)
        
        if model_path is None:
            st.error(f"Failed to download {model_name} from Google Drive")
            return None
        
        # Step 2: Create model architecture
        if model_name == "EfficientNetB0":
            model = models.efficientnet_b0(weights=None)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_features, 4)
            )
        
        elif model_name == "VGG16":
            model = models.vgg16(weights=None)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, 4)
        
        else:
            st.error(f"Unknown model: {model_name}")
            return None
        
        # Step 3: Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Step 4: Load state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        # Step 5: Set to evaluation mode
        model.eval()
        
        st.success(f"✅ {model_name} loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading {model_name}: {str(e)}")
        return None