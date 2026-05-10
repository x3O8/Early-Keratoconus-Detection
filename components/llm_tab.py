"""
LLM Explanation Tab — Gemini-powered clinical report generation.
"""

import streamlit as st
from utils.gemini_client import generate_clinical_explanation


def render_llm_tab():
    ensemble = st.session_state.get("ensemble_result")
    image_result = st.session_state.get("image_result")
    tabular_result = st.session_state.get("tabular_result")

    if not ensemble:
        st.markdown("""
        <div class="kc-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🧠</div>
            <div style="color:#8896b0;">Run analysis first to generate an LLM-powered clinical explanation.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Quick summary cards ──────────────────────────────────────────────────
    kc_prob   = ensemble.get("keratoconus_prob", 0.0)
    severity  = ensemble.get("severity", "—")
    diagnosis = ensemble.get("predicted_label", "—")

    st.markdown(f"""
    <div class="kc-card" style="display:flex; gap:2rem; align-items:center; flex-wrap:wrap;">
        <div>
            <div class="section-label" style="margin-bottom:0.25rem;">DIAGNOSIS</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.1rem; color:#e8edf5;">{diagnosis}</div>
        </div>
        <div>
            <div class="section-label" style="margin-bottom:0.25rem;">KC PROBABILITY</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.1rem; color:#ff4b6e;">{kc_prob:.1%}</div>
        </div>
        <div>
            <div class="section-label" style="margin-bottom:0.25rem;">SEVERITY</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.1rem; color:#ffb300;">{severity}</div>
        </div>
        <div style="flex:1; text-align:right;">
            <div style="font-size:0.75rem; color:#8896b0;">Explanation powered by Gemini 1.5 Flash</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Check API key ────────────────────────────────────────────────────────
    api_key = st.session_state.get("gemini_api_key", "")
    if not api_key:
        st.markdown("""
        <div class="warning-banner">
            🔑 Enter your Gemini API key in the sidebar to generate a clinical explanation.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="kc-card" style="margin-top:1rem;">
            <div class="section-label">How to get a Gemini API Key</div>
            <ol style="color:#8896b0; font-size:0.85rem; line-height:2;">
                <li>Visit <a href="https://aistudio.google.com" target="_blank" style="color:#00e5ff;">aistudio.google.com</a></li>
                <li>Sign in with a Google account</li>
                <li>Click <b>Get API key</b> → <b>Create API key</b></li>
                <li>Copy and paste the key into the sidebar field</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Cached explanation ───────────────────────────────────────────────────
    existing_explanation = st.session_state.get("gemini_explanation")

    col_btn, col_regen = st.columns([2, 1])
    with col_btn:
        generate_btn = st.button("🧠 Generate Clinical Explanation", type="primary", use_container_width=True)
    with col_regen:
        if existing_explanation:
            regen_btn = st.button("🔄 Regenerate", use_container_width=True)
        else:
            regen_btn = False

    if generate_btn or regen_btn:
        feature_importance = st.session_state.get("feature_importance")
        modality_weights = (image_result or {}).get("modality_weights")

        with st.spinner("🧠 Consulting Gemini for clinical reasoning…"):
            explanation = generate_clinical_explanation(
                api_key=api_key,
                ensemble_result=ensemble,
                image_result=image_result or {},
                tabular_result=tabular_result or {},
                feature_importance=feature_importance,
                modality_weights=modality_weights,
            )
            if explanation:
                st.session_state["gemini_explanation"] = explanation

    # ── Display explanation ──────────────────────────────────────────────────
    explanation = st.session_state.get("gemini_explanation")
    if explanation:
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">📋 Clinical Decision Support Report</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kc-card-accent" style="padding:1.75rem; line-height:1.75; color:#d0dbe8; font-size:0.9rem;">
            {_md_to_html(explanation)}
        </div>
        """, unsafe_allow_html=True)

        # Copy button
        st.download_button(
            label="📥 Download Report",
            data=explanation,
            file_name="keratoscan_clinical_report.md",
            mime="text/markdown",
        )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer" style="margin-top:1.5rem;">
        <b>⚠️ Important Disclaimer</b><br>
        This AI-generated clinical report is for research and educational purposes only.
        It is <b>not</b> a medical diagnosis. All findings must be independently verified
        by a qualified ophthalmologist before any clinical decisions are made.
        The AI system may produce errors or incomplete assessments.
    </div>
    """, unsafe_allow_html=True)


def _md_to_html(text: str) -> str:
    """Minimal markdown to HTML conversion for display inside st.markdown."""
    import re

    # Headers
    text = re.sub(r"^## (.+)$", r"<h3 style='color:#00e5ff; font-family:DM Serif Display,serif; margin:1rem 0 0.4rem;'>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^### (.+)$", r"<h4 style='color:#1de9b6; margin:0.75rem 0 0.3rem;'>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$",  r"<h2 style='color:#00e5ff; font-family:DM Serif Display,serif;'>\1</h2>", text, flags=re.MULTILINE)

    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#e8edf5;'>\1</b>", text)

    # Bullets
    text = re.sub(r"^[\-\*] (.+)$", r"<li style='margin:0.25rem 0;'>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"(<li.*?</li>\n?)+", lambda m: f"<ul style='margin:0.5rem 0;'>{m.group()}</ul>", text, flags=re.DOTALL)

    # Newlines
    text = re.sub(r"\n\n", "<br><br>", text)
    text = re.sub(r"\n", "<br>", text)

    return text
