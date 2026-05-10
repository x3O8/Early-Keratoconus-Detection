"""
Predictions Tab — final diagnosis, confidence, ensemble breakdown.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


CLASS_NAMES = ["Normal", "Keratoconus"]
SEVERITY_COLORS = {
    "Unlikely":   "#1de9b6",
    "Borderline": "#ffb300",
    "Mild":       "#ff8c00",
    "Moderate":   "#ff6b35",
    "Severe":     "#ff4b6e",
}


def render_predictions_tab():
    ensemble = st.session_state.get("ensemble_result")
    image_result = st.session_state.get("image_result")
    tabular_result = st.session_state.get("tabular_result")

    if not ensemble:
        st.markdown("""
        <div class="kc-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎯</div>
            <div style="color:#8896b0;">Upload data and run analysis to see predictions.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    kc_prob   = ensemble["keratoconus_prob"]
    severity  = ensemble["severity"]
    diagnosis = ensemble["predicted_label"]
    confidence = ensemble["confidence"]
    sev_color = SEVERITY_COLORS.get(severity, "#8896b0")

    is_kc = (diagnosis == "Keratoconus")
    badge_class = "badge-positive" if is_kc else "badge-negative"

    # ── Top banner ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kc-card-accent" style="text-align:center; padding:2rem 1.5rem;">
        <div class="section-label" style="text-align:center;">ENSEMBLE DIAGNOSIS</div>
        <span class="{badge_class}" style="font-size:1.2rem; padding:0.6rem 2rem;">
            {diagnosis.upper()}
        </span>
        <div style="margin-top:1rem; font-family:'DM Mono',monospace; font-size:2.5rem; color:{sev_color};">
            {kc_prob:.1%}
        </div>
        <div style="color:#8896b0; font-size:0.88rem;">Keratoconus Probability</div>
        <div style="margin-top:0.5rem; display:inline-block; padding:0.25rem 1rem;
                    border:1px solid {sev_color}44; border-radius:12px; color:{sev_color}; font-size:0.85rem;">
            Severity: {severity}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "NORMAL PROB", f"{ensemble['normal_prob']:.1%}", "#1de9b6")
    _metric_card(c2, "KC PROB",     f"{kc_prob:.1%}",               "#ff4b6e" if is_kc else "#8896b0")
    _metric_card(c3, "CONFIDENCE",  f"{confidence:.1%}",            "#00e5ff")
    _metric_card(c4, "SEVERITY",    severity,                        sev_color)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Gauge chart ──────────────────────────────────────────────────────────
    g_col, e_col = st.columns([1, 1])
    with g_col:
        st.markdown('<div class="section-label">Confidence Gauge</div>', unsafe_allow_html=True)
        fig_gauge = _make_gauge(kc_prob, diagnosis, sev_color)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with e_col:
        st.markdown('<div class="section-label">Ensemble Contribution</div>', unsafe_allow_html=True)
        fig_ens = _make_ensemble_bar(ensemble)
        st.plotly_chart(fig_ens, use_container_width=True, config={"displayModeBar": False})

    # ── Per-model breakdown ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Model-wise Predictions</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        _model_result_card("🧠 Image Model", image_result, ensemble["image_weight"])
    with m2:
        _model_result_card("📊 Tabular Model", tabular_result, ensemble["tabular_weight"])

    # ── Class probability bars ───────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Class Probability Distribution</div>', unsafe_allow_html=True)
    fig_bars = _make_prob_bar(image_result, tabular_result, ensemble)
    st.plotly_chart(fig_bars, use_container_width=True, config={"displayModeBar": False})


# ── Helper widgets ───────────────────────────────────────────────────────────

def _metric_card(col, label, value, color):
    with col:
        st.markdown(f"""
        <div class="kc-card" style="text-align:center; padding:1rem 0.5rem;">
            <div class="metric-label">{label}</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.4rem; color:{color};">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def _make_gauge(prob, label, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>{label}</b>", "font": {"color": "#8896b0", "size": 14}},
        number={"suffix": "%", "font": {"color": color, "size": 28, "family": "DM Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#3a4a60", "tickfont": {"color": "#8896b0"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#111622",
            "borderwidth": 1,
            "bordercolor": "#1e2d45",
            "steps": [
                {"range": [0, 40],  "color": "#0d1a28"},
                {"range": [40, 60], "color": "#1a2030"},
                {"range": [60, 100],"color": "#2a1828"},
            ],
            "threshold": {
                "line": {"color": "rgba(255,255,255,0.27)", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=40, b=0, l=20, r=20),
        font={"color": "#e8edf5"},
    )
    return fig


def _make_ensemble_bar(ensemble):
    labels = ["Image\nContribution", "Tabular\nContribution"]
    values = [ensemble["image_contribution"], ensemble["tabular_contribution"]]
    colors = ["#00e5ff", "#1de9b6"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont={"color": "#e8edf5", "size": 13},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=20, b=20, l=20, r=20),
        yaxis={"showgrid": False, "visible": False},
        xaxis={"tickfont": {"color": "#8896b0"}},
        showlegend=False,
    )
    return fig


def _model_result_card(title, result, weight):
    if result is None:
        st.markdown(f"""
        <div class="kc-card">
            <div style="font-weight:600; color:#e8edf5; margin-bottom:0.5rem;">{title}</div>
            <div style="color:#8896b0; font-size:0.85rem;">No data available</div>
        </div>
        """, unsafe_allow_html=True)
        return

    kc_p = result.get("keratoconus_prob", 0.0)
    pred = result.get("predicted_label", "—")
    conf = result.get("confidence", 0.0)
    is_kc = pred == "Keratoconus"
    badge = "badge-positive" if is_kc else "badge-negative"

    st.markdown(f"""
    <div class="kc-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
            <span style="font-weight:600; color:#e8edf5;">{title}</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#8896b0;">weight: {weight:.0%}</span>
        </div>
        <span class="{badge}">{pred.upper()}</span>
        <div style="margin-top:0.75rem; display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
            <div>
                <div style="font-size:0.7rem; color:#8896b0;">KC PROB</div>
                <div style="font-family:'DM Mono',monospace; color:#ff4b6e;">{kc_p:.1%}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; color:#8896b0;">CONFIDENCE</div>
                <div style="font-family:'DM Mono',monospace; color:#00e5ff;">{conf:.1%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _make_prob_bar(image_result, tabular_result, ensemble):
    models = []
    normals = []
    kcs = []

    for name, r in [("Image Model", image_result), ("Tabular Model", tabular_result), ("Ensemble", ensemble)]:
        if r:
            models.append(name)
            normals.append(r.get("normal_prob", 0))
            kcs.append(r.get("keratoconus_prob", 0))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Normal", x=models, y=normals,
        marker_color="#1de9b6", marker_line_width=0,
        text=[f"{v:.1%}" for v in normals], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Keratoconus", x=models, y=kcs,
        marker_color="#ff4b6e", marker_line_width=0,
        text=[f"{v:.1%}" for v in kcs], textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(t=10, b=10, l=10, r=10),
        legend={"font": {"color": "#8896b0"}, "bgcolor": "rgba(0,0,0,0)"},
        yaxis={"showgrid": False, "visible": False},
        xaxis={"tickfont": {"color": "#8896b0"}},
    )
    return fig
