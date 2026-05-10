"""
Explainability Tab — Grad-CAM heatmaps, modality importance, feature importance.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import io
import cv2

MODALITIES = ["CT_A", "EC_A", "EC_P", "Elv_A", "Elv_P", "Sag_A", "Sag_P"]

MODALITY_FULL = {
    "CT_A":  "Corneal Thickness (Ant.)",
    "EC_A":  "Epithelial Curv. (Ant.)",
    "EC_P":  "Epithelial Curv. (Post.)",
    "Elv_A": "Elevation (Ant.)",
    "Elv_P": "Elevation (Post.)",
    "Sag_A": "Sagittal Curv. (Ant.)",
    "Sag_P": "Sagittal Curv. (Post.)",
}


def render_explainability_tab():
    ensemble = st.session_state.get("ensemble_result")
    if not ensemble:
        st.markdown("""
        <div class="kc-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🔬</div>
            <div style="color:#8896b0;">Run analysis first to view explainability outputs.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    image_result = st.session_state.get("image_result")
    gradcam_maps = st.session_state.get("gradcam_maps", {})
    feature_importance = st.session_state.get("feature_importance")
    uploaded_images = st.session_state.get("uploaded_images", {})

    exp_tab1, exp_tab2, exp_tab3 = st.tabs([
        "🌡️ Grad-CAM Heatmaps",
        "📡 Modality Importance",
        "📈 Feature Importance",
    ])

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    with exp_tab1:
        st.markdown('<div class="section-label">Gradient-weighted Class Activation Maps</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#8896b0; font-size:0.82rem; margin-bottom:1rem;">Red/warm regions indicate areas most influential for the prediction. Each map corresponds to a corneal imaging modality.</div>', unsafe_allow_html=True)

        available_mods = [m for m in MODALITIES if uploaded_images.get(m) is not None]
        if not available_mods:
            st.info("No images uploaded — Grad-CAM requires image inputs.")
        else:
            cols_per_row = 4
            for row_start in range(0, len(available_mods), cols_per_row):
                row_mods = available_mods[row_start:row_start + cols_per_row]
                cols = st.columns(len(row_mods))
                for col, mod in zip(cols, row_mods):
                    with col:
                        st.markdown(f"""
                        <div style="font-family:'DM Mono',monospace; font-size:0.75rem;
                                    color:#00e5ff; text-align:center; margin-bottom:0.25rem;">
                            {mod}
                        </div>""", unsafe_allow_html=True)

                        cam = gradcam_maps.get(mod)
                        orig_tensor = uploaded_images.get(mod)

                        if cam is not None and orig_tensor is not None:
                            from utils.preprocessor import tensor_to_display
                            from utils.inference import overlay_gradcam
                            orig_np = tensor_to_display(orig_tensor)
                            overlay = overlay_gradcam(orig_np, cam)
                            st.image(overlay, use_container_width=True, caption="Grad-CAM")
                        elif orig_tensor is not None:
                            from utils.preprocessor import tensor_to_display
                            orig_np = tensor_to_display(orig_tensor)
                            st.image(orig_np, use_container_width=True, caption="Original")
                            st.markdown('<div style="font-size:0.65rem; color:#8896b0; text-align:center;">CAM unavailable</div>', unsafe_allow_html=True)

        # CAM colour scale legend
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.75rem; font-size:0.75rem; color:#8896b0;">
            <span>Low influence</span>
            <div style="flex:1; height:8px; border-radius:4px;
                background:linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000);"></div>
            <span>High influence</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Modality Importance ───────────────────────────────────────────────────
    with exp_tab2:
        st.markdown('<div class="section-label">Cross-Modal Attention Weights</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#8896b0; font-size:0.82rem; margin-bottom:1rem;">The transformer\'s attention mechanism assigns importance weights to each modality. Higher weight → more influential in the final decision.</div>', unsafe_allow_html=True)

        mod_weights = None
        if image_result:
            mod_weights = image_result.get("modality_weights", {})

        if mod_weights:
            fig = _make_modality_chart(mod_weights)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Top modality callout
            top_mod = max(mod_weights, key=mod_weights.get)
            top_val = mod_weights[top_mod]
            st.markdown(f"""
            <div class="kc-card" style="display:flex; align-items:center; gap:1rem;">
                <div style="font-size:2rem;">🏆</div>
                <div>
                    <div style="font-size:0.72rem; color:#8896b0; text-transform:uppercase;">Most Influential Modality</div>
                    <div style="font-family:'DM Mono',monospace; color:#00e5ff; font-size:1.1rem;">{top_mod} — {MODALITY_FULL.get(top_mod, top_mod)}</div>
                    <div style="color:#8896b0; font-size:0.82rem;">Attention weight: {top_val:.1%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Modality weights not available — requires image model results.")

    # ── Feature Importance ────────────────────────────────────────────────────
    with exp_tab3:
        st.markdown('<div class="section-label">Clinical Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#8896b0; font-size:0.82rem; margin-bottom:1rem;">Feature importance scores from the tabular model. Higher values indicate greater discriminative power for the keratoconus prediction.</div>', unsafe_allow_html=True)

        if feature_importance:
            fig = _make_feature_chart(feature_importance)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Top 5 table
            top5 = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
            st.markdown("**Top 5 Features**")
            for rank, (feat, imp) in enumerate(top5, 1):
                bar_width = int(imp * 300)
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;">
                    <span style="font-family:'DM Mono',monospace; color:#8896b0; width:1.5rem; font-size:0.75rem;">#{rank}</span>
                    <span style="color:#e8edf5; width:200px; font-size:0.82rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{feat}</span>
                    <div style="flex:1; background:#1e2d45; border-radius:3px; height:6px;">
                        <div style="width:{min(bar_width,300)}px; max-width:100%; height:6px; border-radius:3px;
                                    background:linear-gradient(90deg,#00e5ff,#1de9b6);"></div>
                    </div>
                    <span style="font-family:'DM Mono',monospace; color:#00e5ff; font-size:0.8rem; width:50px;">{imp:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Feature importance not available — requires tabular model results.")


# ── Chart builders ────────────────────────────────────────────────────────────

def _make_modality_chart(mod_weights):
    mods = list(mod_weights.keys())
    weights = list(mod_weights.values())
    colors = [f"rgba({int(255*(1-w))},{int(200*w+55)},{int(255*w)}, 0.85)" for w in weights]

    fig = go.Figure(go.Bar(
        x=[MODALITY_FULL.get(m, m) for m in mods],
        y=weights,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{w:.1%}" for w in weights],
        textposition="outside",
        textfont={"color": "#e8edf5", "size": 12},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(t=30, b=80, l=20, r=20),
        xaxis={"tickangle": -30, "tickfont": {"color": "#8896b0", "size": 11}},
        yaxis={"showgrid": False, "visible": False},
        showlegend=False,
    )
    return fig


def _make_feature_chart(feat_imp, top_n=20):
    items = sorted(feat_imp.items(), key=lambda x: -x[1])[:top_n]
    feats = [i[0] for i in items]
    imps  = [i[1] for i in items]

    # Gradient colors by rank
    n = len(feats)
    colors = [f"rgba(0,{int(229 * (1 - i/n) + 29 * (i/n))},{int(255 * (1-i/n) + 182*(i/n))},0.85)" for i in range(n)]

    fig = go.Figure(go.Bar(
        y=feats[::-1],
        x=imps[::-1],
        orientation="h",
        marker_color=colors[::-1],
        marker_line_width=0,
        text=[f"{v:.1%}" for v in imps[::-1]],
        textposition="outside",
        textfont={"color": "#e8edf5", "size": 10},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(300, 20 * n + 60),
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis={"showgrid": False, "visible": False},
        yaxis={"tickfont": {"color": "#8896b0", "size": 10}},
        showlegend=False,
    )
    return fig
