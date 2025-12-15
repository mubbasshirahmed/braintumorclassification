"""
Brain Tumor Classification with Explainable AI
COMPLETELY CLEAN VERSION - Zero Emojis, All Flaticon Icons

Author: Mubbasshir
Description: Professional UI with only Flaticon icons
"""

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import time

# Import from our modules
from config import CLASS_NAMES, MODEL_ACCURACY, XAI_ACCURACY, SUPPORTED_IMAGE_FORMATS
from models import model_loader, predictor
from utils.image_processing import preprocess_image
from utils.visualization import create_probability_chart
from xai.gradcam import GradCAM, apply_gradcam_overlay
from xai.lime_explainer import generate_lime_explanation
from xai.lrp import generate_lrp_explanation, apply_lrp_overlay


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Brain Tumor AI Classifier",
    page_icon="https://cdn-icons-png.flaticon.com/32/2382/2382461.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS - FORCE SIDEBAR VISIBLE, NO COLLAPSE
# ============================================================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* FORCE SIDEBAR TO BE ALWAYS VISIBLE - NO COLLAPSE BUTTON */
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    
    /* Force sidebar to stay open */
    [data-testid="stSidebar"] {
        position: relative !important;
        transform: none !important;
        transition: none !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 280px !important;
        max-width: 280px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 280px !important;
        max-width: 280px !important;
        display: block !important;
        visibility: visible !important;
    }
    
    /* Remove padding at top */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        color: #1e293b;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    .main-header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1rem;
        text-align: center;
        color: #475569;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Icons */
    .header-icon {
        width: 48px;
        height: 48px;
    }
    
    .section-icon {
        width: 28px;
        height: 28px;
        margin-right: 0.5rem;
    }
    
    .small-icon {
        width: 16px;
        height: 16px;
        margin-right: 0.3rem;
        vertical-align: middle;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.95;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    
    /* Success Card */
    .success-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
        margin: 1.5rem 0;
    }
    
    .success-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .success-card h4 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .info-box p {
        margin: 0;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
    }
    
    /* Image Container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin: 1rem 0;
        background: white;
        padding: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4c63d2 0%, #6b46c1 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Text Colors */
    .main * {
        color: #1e293b;
    }
    
    .success-card *, .info-box *, .metric-card * {
        color: white !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown('''
<div style="text-align: center; margin-bottom: 2rem;">
    <div class="main-header">
        <img src="https://cdn-icons-png.flaticon.com/512/2382/2382461.png" 
             class="header-icon" 
             alt="Brain Icon" />
        <span class="main-header-gradient">Brain Tumor AI Classifier</span>
    </div>
    <p class="sub-header">
        Advanced AI-powered diagnosis using Transfer Learning & Explainable AI
    </p>
</div>
''', unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <img src="https://cdn-icons-png.flaticon.com/512/2382/2382461.png" 
             style="width: 80px; margin-bottom: 1rem;" 
             alt="Brain" />
        <h2 style='margin: 0; font-size: 1.5rem;'>Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    st.markdown("**Select Model**")
    model_choice = st.selectbox(
        "Model",
        ["EfficientNetB0"],
        help="Choose between EfficientNetB0 (95.2%) and VGG16 (93.8%)",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # XAI method selection
    st.markdown("**XAI Visualization**")
    xai_method = st.selectbox(
        "XAI",
        ["Grad-CAM", "LIME", "LRP"],
        help="Select explainability technique",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System info with icons
    st.markdown("### System Info")
    st.markdown(f"""
    <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" class="small-icon" /> Models: 2  
    <img src="https://cdn-icons-png.flaticon.com/512/2920/2920277.png" class="small-icon" /> Classes: {len(CLASS_NAMES)}  
    <img src="https://cdn-icons-png.flaticon.com/512/1995/1995467.png" class="small-icon" /> XAI Methods: 3
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    <img src="https://cdn-icons-png.flaticon.com/512/3004/3004457.png" class="small-icon" /> Transfer Learning  
    <img src="https://cdn-icons-png.flaticon.com/512/1995/1995467.png" class="small-icon" /> Explainable AI  
    <img src="https://cdn-icons-png.flaticon.com/512/2382/2382461.png" class="small-icon" /> Deep Learning  
    <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" class="small-icon" /> Medical Imaging
    """, unsafe_allow_html=True)


# ============================================================
# MAIN CONTENT
# ============================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('''
    <div class="section-header">
        <img src="https://cdn-icons-png.flaticon.com/512/1533/1533913.png" 
             class="section-icon" 
             alt="Upload" />
        Upload MRI Scan
    </div>
    ''', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop your MRI image here",
        type=SUPPORTED_IMAGE_FORMATS,
        help="Supported formats: JPG, JPEG, PNG (Max 200MB)",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption='Uploaded MRI Scan', width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div class="section-header">
        <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" 
             class="section-icon" 
             alt="Results" />
        Classification Results
    </div>
    ''', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        if st.button("Analyze Tumor", type="primary", use_container_width=True):
            
            # Progress
            with st.spinner(f'Loading {model_choice} model...'):
                progress_bar = st.progress(0)
                for i in range(30):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                model = model_loader.load_model(model_choice)
                progress_bar.progress(100)
            
            if model is not None:
                with st.spinner("Processing MRI scan..."):
                    image_tensor = preprocess_image(image)
                    result = predictor.predict(model, image_tensor)
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['model_name'] = model_choice
                st.session_state['image'] = image
                st.session_state['image_tensor'] = image_tensor
                st.session_state['result'] = result
                
                # Display results

                
                st.markdown(f"""
                <div class="success-card">
                    <h3>Predicted: {result['label']}</h3>
                    <h4>Confidence: {result['confidence']*100:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                st.markdown('''
                <div class="section-header">
                    <img src="https://cdn-icons-png.flaticon.com/512/2920/2920277.png" 
                         class="section-icon" 
                         alt="Chart" />
                    Probability Distribution
                </div>
                ''', unsafe_allow_html=True)
                
                fig = create_probability_chart(
                    result['probabilities'],
                    result['class'],
                    model_choice
                )
                st.pyplot(fig)
                plt.close()


# ============================================================
# XAI VISUALIZATION SECTION
# ============================================================
if uploaded_file is not None and 'result' in st.session_state:
    st.markdown("---")
    st.markdown(f'''
    <div class="section-header">
        <img src="https://cdn-icons-png.flaticon.com/512/1995/1995467.png" 
             class="section-icon" 
             alt="XAI" />
        Explainable AI: {xai_method}
    </div>
    ''', unsafe_allow_html=True)
    
    col3, col4 = st.columns([1, 1], gap="large")
    
    with col3:
        st.markdown("### Original MRI")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(st.session_state['image'], width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if st.button(f"Generate {xai_method}", type="secondary", use_container_width=True):
            model = st.session_state['model']
            model_name = st.session_state['model_name']
            image = st.session_state['image']
            image_tensor = st.session_state['image_tensor']
            target_class = st.session_state['result']['class']
            
            if xai_method == "Grad-CAM":
                with st.spinner("Generating Grad-CAM..."):
                    gradcam = GradCAM(model, model_name)
                    heatmap = gradcam.generate(image_tensor, target_class)
                    overlay = apply_gradcam_overlay(image, heatmap)
                
                st.markdown("### Grad-CAM Heatmap")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(overlay, caption=f"Important regions for {CLASS_NAMES[target_class]}", width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    <p><strong>Grad-CAM</strong> highlights regions important for prediction.</p>
                    <p> 
                        Red/Yellow = High importance |  
                        Blue = Low importance
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif xai_method == "LIME":
                with st.spinner("Generating LIME (10-20 seconds)..."):
                    lime_img = generate_lime_explanation(model, image, target_class)
                
                st.markdown("### LIME Superpixels")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(lime_img, caption=f"Superpixel explanation for {CLASS_NAMES[target_class]}", width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    <p><strong>LIME</strong> shows which superpixels contribute to prediction.</p>
                    <p>Highlighted boundaries indicate important regions.</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif xai_method == "LRP":
                with st.spinner("Generating LRP..."):
                    relevance_map = generate_lrp_explanation(model, image_tensor, target_class)
                    overlay = apply_lrp_overlay(image, relevance_map)
                
                st.markdown("### LRP Relevance")
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(overlay, caption=f"Pixel relevance for {CLASS_NAMES[target_class]}", width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    <p><strong>LRP</strong> shows pixel-wise relevance scores.</p>
                    <p> 
                        Warmer colors = Higher relevance
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# METRICS FOOTER
# ============================================================
st.markdown("---")
st.markdown('''
<div class="section-header">
    <img src="https://cdn-icons-png.flaticon.com/512/2920/2920277.png" 
         class="section-icon" 
         alt="Metrics" />
    Performance Metrics
</div>
''', unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)
col_m4, col_m5 = st.columns(2)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">EfficientNetB0</div>
        <div class="metric-value">{MODEL_ACCURACY['EfficientNetB0']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">VGG16</div>
        <div class="metric-value">{MODEL_ACCURACY['VGG16']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Grad-CAM</div>
        <div class="metric-value">{XAI_ACCURACY['Grad-CAM']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">LIME</div>
        <div class="metric-value">{XAI_ACCURACY['LIME']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">LRP</div>
        <div class="metric-value">{XAI_ACCURACY['LRP']}</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 1.5rem 0;'>
    <p style='font-size: 0.9rem; margin: 0;'>
        <img src="https://cdn-icons-png.flaticon.com/512/2382/2382461.png" 
             style="width: 20px; vertical-align: middle; margin-right: 0.5rem;" />
        Brain Tumor AI Classifier | Transfer Learning & Explainable AI
    </p>
</div>
""", unsafe_allow_html=True)

# JavaScript to force sidebar always visible
st.markdown("""
<script>
    // Force sidebar to be visible on page load
    window.addEventListener('load', function() {
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
            sidebar.style.transform = 'none';
            sidebar.setAttribute('aria-expanded', 'true');
        }
        
        // Hide collapse button
        const collapseBtn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
        if (collapseBtn) {
            collapseBtn.style.display = 'none';
        }
    });
    
    // Check every second to ensure sidebar stays visible
    setInterval(function() {
        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
            sidebar.setAttribute('aria-expanded', 'true');
        }
    }, 1000);
</script>
""", unsafe_allow_html=True)