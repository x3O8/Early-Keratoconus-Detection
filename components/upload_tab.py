"""
Upload Tab — image and CSV upload with previews and validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from utils.preprocessor import preprocess_image, validate_csv

MODALITIES = ["CT_A", "EC_A", "EC_P", "Elv_A", "Elv_P", "Sag_A", "Sag_P"]

MODALITY_DESCRIPTIONS = {
    "CT_A":  "Corneal Thickness — Anterior",
    "EC_A":  "Epithelial Curvature — Anterior",
    "EC_P":  "Epithelial Curvature — Posterior",
    "Elv_A": "Elevation Map — Anterior",
    "Elv_P": "Elevation Map — Posterior",
    "Sag_A": "Sagittal Curvature — Anterior",
    "Sag_P": "Sagittal Curvature — Posterior",
}


def render_upload_tab():
    st.markdown('<div class="section-label">📤 Data Input</div>', unsafe_allow_html=True)

    # Ensure models loaded
    if not st.session_state.get("models_loaded"):
        st.markdown("""
        <div class="warning-banner">
            ⚠️ Models not loaded. Click <b>Load Models</b> in the sidebar before uploading data.
        </div>
        """, unsafe_allow_html=True)

    upload_col, info_col = st.columns([3, 1])

    with info_col:
        st.markdown("""
        <div class="kc-card" style="height:100%;">
            <div class="section-label">📋 Requirements</div>
            <div style="font-size:0.82rem; color:#8896b0; line-height:1.8;">
                <b style="color:#e8edf5;">Images</b><br>
                • JPG / PNG / JPEG<br>
                • All 7 modalities recommended<br>
                • Missing modalities → zero-filled<br><br>
                <b style="color:#e8edf5;">CSV</b><br>
                • Header row + 1 data row<br>
                • All numeric features<br>
                • Must match model columns
            </div>
        </div>
        """, unsafe_allow_html=True)

    with upload_col:
        # ── Image Uploads ─────────────────────────────────────────────────
        st.markdown("#### 🖼️ Corneal Image Modalities")

        uploaded_images = st.session_state.get("uploaded_images", {})
        row1 = MODALITIES[:4]
        row2 = MODALITIES[4:]

        def upload_row(mods):
            cols = st.columns(len(mods))
            for col, mod in zip(cols, mods):
                with col:
                    st.markdown(f"""
                    <div style="font-size:0.72rem; font-family:'DM Mono',monospace; color:#00e5ff;
                                margin-bottom:0.25rem; text-align:center;">{mod}</div>
                    <div style="font-size:0.65rem; color:#8896b0; text-align:center;
                                margin-bottom:0.5rem;">{MODALITY_DESCRIPTIONS[mod]}</div>
                    """, unsafe_allow_html=True)
                    f = st.file_uploader(
                        f"Upload {mod}",
                        type=["jpg", "png", "jpeg"],
                        key=f"upload_{mod}",
                        label_visibility="collapsed",
                    )
                    if f is not None:
                        img = Image.open(io.BytesIO(f.read()))
                        st.image(img, use_container_width=True)
                        f.seek(0)
                        tensor = preprocess_image(f)
                        uploaded_images[mod] = tensor
                        st.markdown(f'<div style="text-align:center; font-size:0.7rem; color:#1de9b6;">✓ {img.size[0]}×{img.size[1]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="text-align:center; font-size:0.7rem; color:#3a4a60; padding:0.5rem;">No file</div>', unsafe_allow_html=True)

        upload_row(row1)
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        upload_row(row2)

        st.session_state["uploaded_images"] = uploaded_images

        # Summary
        n_uploaded = sum(1 for v in uploaded_images.values() if v is not None)
        total = len(MODALITIES)
        progress = n_uploaded / total

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        st.progress(progress, text=f"Modalities uploaded: {n_uploaded}/{total}")

        if n_uploaded < total:
            missing = [m for m in MODALITIES if uploaded_images.get(m) is None]
            st.markdown(f'<div class="warning-banner">Missing modalities will be zero-filled: {", ".join(missing)}</div>', unsafe_allow_html=True)

        st.divider()

        # ── CSV Upload ────────────────────────────────────────────────────
        st.markdown("#### 📊 Clinical/Topography Data (CSV)")

        csv_file = st.file_uploader(
            "Upload patient CSV",
            type=["csv"],
            key="csv_uploader",
            help="One row per patient. First row must be column headers.",
        )

        if csv_file is not None:
            tabular_model = st.session_state.get("tabular_model")

            df, errors, warnings = validate_csv(csv_file, tabular_model)

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
                st.session_state["uploaded_csv"] = None
            else:
                st.session_state["uploaded_csv"] = df

                for warn in warnings:
                    st.warning(f"⚠️ {warn}")

                st.markdown("**Preview — Patient Record**")
                # Transpose for readability
                display_df = df.T.reset_index()
                display_df.columns = ["Feature", "Value"]
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=min(300, 36 + 35 * len(display_df)),
                    hide_index=True,
                )
                st.markdown(f'<div style="color:#1de9b6; font-size:0.82rem;">✅ {df.shape[1]} features loaded</div>', unsafe_allow_html=True)
        else:
            st.session_state["uploaded_csv"] = None

    st.divider()

    # ── Run Inference Button ─────────────────────────────────────────────
    can_run = (
        st.session_state.get("models_loaded", False)
        and (
            any(v is not None for v in st.session_state.get("uploaded_images", {}).values())
            or st.session_state.get("uploaded_csv") is not None
        )
    )

    if can_run:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=False):
            _run_full_analysis()
    else:
        st.markdown("""
        <div style="color:#8896b0; font-size:0.85rem;">
            Load models and upload at least one image or CSV to enable analysis.
        </div>
        """, unsafe_allow_html=True)

    # Show status
    if st.session_state.get("ensemble_result"):
        st.success("✅ Analysis complete — view results in the Predictions tab.")


def _run_full_analysis():
    """Run image inference, tabular inference, and ensemble fusion."""
    from utils.inference import (
        run_image_inference, run_tabular_inference,
        ensemble_fusion, compute_gradcam,
        compute_integrated_gradients, compute_attention_rollout,
    )

    model = st.session_state.get("image_model")
    tabular_model = st.session_state.get("tabular_model")
    images = st.session_state.get("uploaded_images", {})
    csv_df = st.session_state.get("uploaded_csv")
    device = st.session_state.get("device", "cpu")

    import torch
    dev = torch.device(device)

    progress_bar = st.progress(0, text="Initialising…")

    # Image inference
    image_result = None
    if model and images:
        progress_bar.progress(0.1, text="Running image model…")
        try:
            image_result = run_image_inference(model, images, dev)
            st.session_state["image_result"] = image_result
        except Exception as e:
            st.error(f"Image inference error: {e}")

    # Tabular inference
    tabular_result = None
    if tabular_model and csv_df is not None:
        progress_bar.progress(0.4, text="Running tabular model…")
        try:
            tabular_result = run_tabular_inference(tabular_model, csv_df)
            st.session_state["tabular_result"] = tabular_result
            st.session_state["feature_importance"] = tabular_result.get("feature_importance")
        except Exception as e:
            st.error(f"Tabular inference error: {e}")

    # Ensemble
    if image_result or tabular_result:
        progress_bar.progress(0.6, text="Fusing predictions…")

        if image_result is None:
            image_result = {"probs": [0.5, 0.5], "keratoconus_prob": 0.5, "normal_prob": 0.5,
                            "predicted_class": 0, "predicted_label": "Normal",
                            "confidence": 0.5, "modality_weights": {}}
        if tabular_result is None:
            tabular_result = {"probs": [0.5, 0.5], "keratoconus_prob": 0.5, "normal_prob": 0.5,
                              "predicted_class": 0, "predicted_label": "Normal",
                              "confidence": 0.5, "feature_importance": {}}

        ensemble_result = ensemble_fusion(
            image_result, tabular_result,
            st.session_state["image_weight"],
            st.session_state["tabular_weight"],
        )
        st.session_state["ensemble_result"] = ensemble_result

    # Visual Explanations (Grad-CAM, Integrated Gradients, Attention Rollout)
    if model and images:
        progress_bar.progress(0.70, text="Computing Grad-CAM…")
        try:
            target_class = st.session_state["ensemble_result"]["predicted_class"]
            gradcam_maps = compute_gradcam(model, images, dev, target_class)
            st.session_state["gradcam_maps"] = gradcam_maps
        except Exception as e:
            st.session_state["gradcam_maps"] = {}

        progress_bar.progress(0.80, text="Computing Integrated Gradients…")
        try:
            target_class = st.session_state["ensemble_result"]["predicted_class"]
            ig_maps = compute_integrated_gradients(model, images, dev, target_class)
            st.session_state["ig_maps"] = ig_maps
        except Exception as e:
            st.session_state["ig_maps"] = {}

        progress_bar.progress(0.90, text="Computing Attention Rollout…")
        try:
            rollout_maps = compute_attention_rollout(model, images, dev)
            st.session_state["rollout_maps"] = rollout_maps
        except Exception as e:
            st.session_state["rollout_maps"] = {}

    progress_bar.progress(1.0, text="Done!")
    import time; time.sleep(0.4)
    progress_bar.empty()

    st.success("✅ Analysis complete!")
    st.rerun()
