"""
Sidebar — model settings, fusion weights, API key, device status.
"""

import streamlit as st
import torch
from utils.model_loader import load_image_model, load_tabular_model, get_device


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
            <div style="font-family:'Roboto',sans-serif; font-size:1.15rem;
                        font-weight:700; color:#00e5ff; letter-spacing:0.03em;">
                Keratoconus Detection
            </div>
            <div style="font-size:0.68rem; color:#8896b0; letter-spacing:0.1em; margin-top:0.25rem;">
                v1.0 · RESEARCH BUILD
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Device Status ────────────────────────────────────────────────────
        device = get_device()
        st.session_state["device"] = str(device)
        device_label = str(device).upper()
        device_color = "#1de9b6" if str(device) != "cpu" else "#00e5ff"
        st.markdown(f"""
        <div style="background:#0d1320; border:1px solid #1e2d45; border-radius:8px;
                    padding:0.6rem 1rem; margin-bottom:1rem;">
            <span style="font-size:0.68rem; color:#8896b0; text-transform:uppercase;
                         letter-spacing:0.1em;">COMPUTE</span><br>
            <span style="font-family:'Roboto Mono',monospace; color:{device_color};
                         font-size:0.9rem;">{device_label}</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Load Models ──────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Model Management</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Models", use_container_width=True):
                with st.spinner("Loading models…"):
                    try:
                        model, dev, img_loaded, img_path = load_image_model()
                        tab_model, tab_loaded, tab_path = load_tabular_model()
                        st.session_state["image_model"] = model
                        st.session_state["tabular_model"] = tab_model
                        st.session_state["models_loaded"] = True
                        if img_loaded and img_path:
                            st.toast(f"Image model: {img_path}")
                        else:
                            st.toast("Image model: demo mode (no .pth found)")
                        if tab_loaded and tab_path:
                            st.toast(f"Tabular model: {tab_path}")
                        else:
                            st.toast("Tabular model: demo mode (no .pkl found)")
                        if img_loaded and tab_loaded:
                            st.success("Both models loaded.")
                        elif img_loaded or tab_loaded:
                            st.warning("Partial load — one model in demo mode.")
                        else:
                            st.info("Demo mode: no checkpoints found.")
                    except Exception as e:
                        st.error(f"Load error: {e}")

        with col2:
            if st.button("Reset", use_container_width=True):
                for k in ["image_result", "tabular_result", "ensemble_result",
                          "gradcam_maps", "feature_importance", "gemini_explanation",
                          "uploaded_images", "uploaded_csv"]:
                    st.session_state[k] = {} if "images" in k or "gradcam" in k else None
                st.rerun()

        # Model status
        img_status = "Loaded" if st.session_state.get("image_model") else "Not loaded"
        tab_status = "Loaded" if st.session_state.get("tabular_model") else "Not loaded"
        img_color  = "#1de9b6" if st.session_state.get("image_model") else "#8896b0"
        tab_color  = "#1de9b6" if st.session_state.get("tabular_model") else "#8896b0"
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#8896b0; margin-top:0.5rem; line-height:1.8;">
            Image model: <span style="color:{img_color};">{img_status}</span><br>
            Tabular model: <span style="color:{tab_color};">{tab_status}</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Ensemble Weights ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">Ensemble Fusion Weights</div>', unsafe_allow_html=True)

        img_w = st.slider(
            "Image Model Weight",
            min_value=0.0, max_value=1.0,
            value=st.session_state["image_weight"],
            step=0.05,
            key="img_weight_slider",
        )
        tab_w = 1.0 - img_w
        st.session_state["image_weight"] = img_w
        st.session_state["tabular_weight"] = tab_w

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div style="text-align:center; background:#0d1320; border-radius:8px; padding:0.5rem;">
                <div style="font-size:0.65rem; color:#8896b0; letter-spacing:0.08em;">IMAGE</div>
                <div style="font-family:'Roboto Mono',monospace; color:#00e5ff; font-size:1.1rem;">{img_w:.0%}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div style="text-align:center; background:#0d1320; border-radius:8px; padding:0.5rem;">
                <div style="font-size:0.65rem; color:#8896b0; letter-spacing:0.08em;">TABULAR</div>
                <div style="font-family:'Roboto Mono',monospace; color:#1de9b6; font-size:1.1rem;">{tab_w:.0%}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Groq API Key (primary LLM) ────────────────────────────────────────
        st.markdown('<div class="section-label">Groq API Key</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#8896b0; font-size:0.75rem; margin-bottom:0.4rem;">'
            'Free account at <a href="https://console.groq.com" target="_blank" '
            'style="color:#00e5ff;">console.groq.com</a> · '
            'No billing required · 14,400 req/day · Llama 3.1 8B</div>',
            unsafe_allow_html=True,
        )
        hf_token = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_…",
            help="Free Groq key. Stored in session only, never logged.",
            label_visibility="collapsed",
            key="hf_token_input",
        )
        if hf_token:
            st.session_state["hf_token"] = hf_token
            st.markdown('<div style="color:#1de9b6; font-size:0.78rem;">Key set (session only)</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#8896b0; font-size:0.75rem;">Enter key to enable live LLM.</div>',
                        unsafe_allow_html=True)

        st.divider()

        # ── Gemini API Key (optional fallback) ───────────────────────────────
        with st.expander("Gemini API Key (optional)"):
            st.markdown('<div style="color:#8896b0; font-size:0.75rem; margin-bottom:0.4rem;">Only needed if you select the Gemini backend in the Clinical Report tab.</div>',
                        unsafe_allow_html=True)
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIza…",
                label_visibility="collapsed",
                key="gemini_key_input",
            )
            if api_key:
                st.session_state["gemini_api_key"] = api_key
                st.markdown('<div style="color:#1de9b6; font-size:0.78rem;">Gemini key set</div>',
                            unsafe_allow_html=True)

        st.divider()

        # ── About ────────────────────────────────────────────────────────────
        with st.expander("About"):
            st.markdown("""
            **Keratoconus Detection** combines:
            - 7-modality corneal image analysis
            - Clinical topography/biometry ML model
            - Weighted ensemble fusion
            - RAG-augmented clinical knowledge retrieval
            - Instant hardcoded clinical reports (demo mode)
            - Optional live LLM via Groq (Llama 3.1) or Gemini

            **Model discovery:** Auto-scans for `.pth` and `.pkl` files.

            *Research tool only — not for clinical use.*
            """)

        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.68rem; color:#3a4a60; text-align:center;">
            Device: {device_label} · Session-scoped
        </div>
        """, unsafe_allow_html=True)
