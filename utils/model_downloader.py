import os
import gdown
import streamlit as st
from pathlib import Path


MODEL_URLS = {
    "EfficientNetB0": "18AkhtrFwInrroKAg0plX2bLF-A9obiMY",  
    "VGG16": "17JGGp6OTLGgFqWBRTvxO87YVCP9EPv-g"  
}


def download_model_from_gdrive(model_name: str) -> str:
    """
    Download model from Google Drive if not exists locally.
    
    Args:
        model_name: "EfficientNetB0" or "VGG16"
        
    Returns:
        Path to downloaded model file
    """
    
    # Construct model path
    if model_name == "EfficientNetB0":
        model_filename = "efficientnet_b0_trained.pth"
    else:
        model_filename = "vgg16_trained.pth"
    
    model_path = f"models/{model_filename}"
    
    # Check if model already exists
    if os.path.exists(model_path):
        return model_path
    
    # Model doesn't exist - download it
    st.info(f"Downloading {model_name} model from Google Drive...")
    st.info("This happens only once and may take 2-3 minutes...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Get file ID
    file_id = MODEL_URLS.get(model_name)
    
    if not file_id or file_id.startswith("YOUR_"):
        st.error(f"Google Drive File ID not configured for {model_name}")
        st.error("Please update MODEL_URLS in utils/model_downloader.py")
        return None
    
    # Construct download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Download with progress bar
        with st.spinner(f"Downloading {model_name}... Please wait..."):
            output = gdown.download(url, model_path, quiet=False)
        
        if output:
            st.success(f"{model_name} downloaded successfully!")
            return model_path
        else:
            st.error(f"Failed to download {model_name}")
            return None
            
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        st.error("Please check:")
        st.error("1. File ID is correct")
        st.error("2. Google Drive link is set to 'Anyone with link can view'")
        st.error("3. Your internet connection is stable")
        return None


def check_models_exist() -> dict:
    """
    Check which models are available locally.
    
    Returns:
        Dictionary with model availability status
    """
    return {
        "EfficientNetB0": os.path.exists("models/efficientnet_b0_trained.pth"),
        "VGG16": os.path.exists("models/vgg16_trained.pth")
    }