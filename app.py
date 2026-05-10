"""
Keratoconus Detection Dashboard
Multi-modal AI system combining ophthalmic image analysis and tabular clinical data.
"""

import streamlit as st
import sys
import os

# Page config MUST be first
st.set_page_config(
    page_title="KeratoScan AI — Keratoconus Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0d14;
    --bg-secondary: #111622;
    --bg-card: #161d2e;
    --accent-cyan: #00e5ff;
    --accent-teal: #1de9b6;
    --accent-red: #ff4b6e;
    --accent-amber: #ffb300;
    --text-primary: #e8edf5;
    --text-secondary: #8896b0;
    --border: #1e2d45;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp { background-color: var(--bg-primary); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1320 0%, #111a2e 100%);
    border-right: 1px solid var(--border);
}

/* Headers */
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #00e5ff, #1de9b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* Cards */
.kc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.kc-card-accent {
    background: linear-gradient(135deg, #0d1d30, #0a1520);
    border: 1px solid #00e5ff33;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result badges */
.badge-positive {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    background: linear-gradient(135deg, #ff4b6e22, #ff4b6e44);
    border: 1px solid #ff4b6e88;
    border-radius: 20px;
    color: #ff4b6e;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
}

.badge-negative {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    background: linear-gradient(135deg, #1de9b622, #1de9b644);
    border: 1px solid #1de9b688;
    border-radius: 20px;
    color: #1de9b6;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: var(--accent-cyan);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: var(--text-secondary);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00e5ff22, #1de9b622);
    border: 1px solid #00e5ff66;
    color: var(--accent-cyan);
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: var(--accent-cyan);
    background: linear-gradient(135deg, #00e5ff33, #1de9b633);
}

/* Info/Warning */
.stAlert { border-radius: 8px; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card);
    border-radius: 8px;
    color: var(--text-primary);
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal));
    border-radius: 4px;
}

/* Divider */
.kc-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Mono text */
.mono { font-family: 'DM Mono', monospace; font-size: 0.85rem; }

.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent-cyan);
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.warning-banner {
    background: #ffb30011;
    border: 1px solid #ffb30044;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #ffb300;
    font-size: 0.88rem;
    margin: 0.5rem 0;
}

.disclaimer {
    background: #161d2e;
    border-left: 3px solid #00e5ff;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

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
    <div class="hero-title">👁️ KeratoScan AI</div>
    <div class="hero-subtitle">
        MULTI-MODAL KERATOCONUS DETECTION SYSTEM &nbsp;·&nbsp;
        OPHTHALMIC IMAGE + CLINICAL DATA FUSION &nbsp;·&nbsp;
        AI-ASSISTED DIAGNOSIS
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ <b>Research & Educational Tool Only.</b> This system is not a substitute for professional medical diagnosis. All results must be reviewed by a qualified ophthalmologist.</div>', unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
render_sidebar()

# ── Main tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📤  Upload Data",
    "🎯  Predictions",
    "🔬  Explainability",
    "🧠  LLM Explanation",
])

with tab1:
    render_upload_tab()

with tab2:
    render_predictions_tab()

with tab3:
    render_explainability_tab()

with tab4:
    render_llm_tab()
