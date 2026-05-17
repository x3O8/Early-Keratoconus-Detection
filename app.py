"""
Keratoconus Detection Dashboard
Multi-modal AI system combining ophthalmic image analysis and tabular clinical data.
"""

import streamlit as st
from utils.ui_components import load_css, render_header, render_sidebar, render_main_tabs
import sys
import os

# Page config MUST be first
st.set_page_config(
    page_title="Keratoconus Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
load_css("assets/style.css")

import pandas as pd
import numpy as np
from PIL import Image
import io

# Internal modules
from utils.model_loader import load_image_model, load_tabular_model, get_device
from utils.preprocessor import preprocess_image, validate_csv
from utils.inference import run_image_inference, run_tabular_inference, ensemble_fusion
from utils.gemini_client import generate_clinical_explanation
from components.sidebar import render_sidebar
from components.upload_tab import render_upload_tab
from components.predictions_tab import render_predictions_tab
from components.explainability_tab import render_explainability_tab
from components.llm_tab import render_llm_tab

# ── Session state defaults ──────────────────────────────────────────────────
defaults = {
    "uploaded_images": {},
    "uploaded_csv": None,
    "image_model": None,
    "tabular_model": None,
    "image_result": None,
    "tabular_result": None,
    "ensemble_result": None,
    "gradcam_maps": {},
    "feature_importance": None,
    "gemini_explanation": None,
    "models_loaded": False,
    "image_weight": 0.65,
    "tabular_weight": 0.35,
    "gemini_api_key": "",
    "device": "cpu",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Hero header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 0.5rem 0;">
    <div class="hero-title">Keratoconus Detection</div>
    <div class="hero-subtitle">
        MULTI-MODAL AI SYSTEM &nbsp;&middot;&nbsp;
        OPHTHALMIC IMAGE + CLINICAL DATA FUSION &nbsp;&middot;&nbsp;
        AI-ASSISTED DIAGNOSIS
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer"><b>Research &amp; Educational Tool Only.</b> This system is not a substitute for professional medical diagnosis. All results must be reviewed by a qualified ophthalmologist.</div>', unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
render_sidebar()

# ── Main tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Upload Data",
    "Predictions",
    "Explainability",
    "Clinical Report",
])

with tab1:
    render_upload_tab()

with tab2:
    render_predictions_tab()

with tab3:
    render_explainability_tab()

with tab4:
    render_llm_tab()
