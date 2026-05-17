import streamlit as st
import os

def load_css(css_path: str):
    """Load a CSS file from the given relative path and inject it into the Streamlit app."""
    full_path = os.path.join(os.getcwd(), css_path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_path}")

def render_header():
    """Render the cinematic hero header used across the app."""
    st.markdown(
        """
        <div style='padding: 1rem 0 0.5rem 0;'>
            <div class='hero-title'>👁️ KeratoScan AI</div>
            <div class='hero-subtitle'>
                MULTI-MODAL KERATOCONUS DETECTION SYSTEM&nbsp;·&nbsp;
                OPHTHALMIC IMAGE + CLINICAL DATA FUSION&nbsp;·&nbsp;
                AI‑ASSISTED DIAGNOSIS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='disclaimer'>⚠️ <b>Research & Educational Tool Only.</b> This system is not a substitute for professional medical diagnosis. All results must be reviewed by a qualified ophthalmologist.</div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar():
    """Delegate sidebar rendering to the existing component for consistency."""
    from components.sidebar import render_sidebar as _render_sidebar
    _render_sidebar()

def render_main_tabs():
    """Create the four main tabs and plug in the respective component renderers."""
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤  Upload Data",
        "🎯  Predictions",
        "🔬  Explainability",
        "🧠  LLM Explanation",
    ])
    with tab1:
        from components.upload_tab import render_upload_tab
        render_upload_tab()
    with tab2:
        from components.predictions_tab import render_predictions_tab
        render_predictions_tab()
    with tab3:
        from components.explainability_tab import render_explainability_tab
        render_explainability_tab()
    with tab4:
        from components.llm_tab import render_llm_tab
        render_llm_tab()
