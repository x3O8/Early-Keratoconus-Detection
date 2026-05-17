"""
LLM Explanation Tab — hardcoded reports for presentation demo.

Two detailed clinical reports pre-written for the two demo cases:
  • Keratoconus  : KC probability ≥ 50%  (demo: 78.9%)
  • Normal       : KC probability <  50%  (demo: 22.9%)

The "Live LLM" option (Groq / Gemini) is still available via the backend selector.
"""

import streamlit as st

from utils.rag_engine import (
    get_rag_index,
    build_rag_context,
    build_query_from_results,
    get_index_stats,
)
from utils.hardcoded_reports import get_hardcoded_report
from utils.hf_client import generate_clinical_explanation_hf
from utils.gemini_client import generate_clinical_explanation as generate_gemini


def render_llm_tab():
    ensemble       = st.session_state.get("ensemble_result")
    image_result   = st.session_state.get("image_result")
    tabular_result = st.session_state.get("tabular_result")

    if not ensemble:
        st.markdown("""
        <div class="kc-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🧠</div>
            <div style="color:#8896b0;">Run analysis first to generate a clinical explanation.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Summary banner ───────────────────────────────────────────────────────
    kc_prob   = ensemble.get("keratoconus_prob", 0.0)
    severity  = ensemble.get("severity", "—")
    diagnosis = ensemble.get("predicted_label", "—")

    sev_color = {
        "Unlikely":   "#1de9b6",
        "Borderline": "#ffb300",
        "Mild":       "#ff8c00",
        "Moderate":   "#ff6b35",
        "Severe":     "#ff4b6e",
    }.get(severity, "#8896b0")

    st.markdown(f"""
    <div class="kc-card" style="display:flex; gap:2rem; align-items:center; flex-wrap:wrap;">
        <div>
            <div class="section-label" style="margin-bottom:0.25rem;">DIAGNOSIS</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.1rem; color:#e8edf5;">{diagnosis}</div>
        </div>
        <div>
            <div class="section-label" style="margin-bottom:0.25rem;">KC PROBABILITY</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.4rem; color:{sev_color};">{kc_prob:.1%}</div>
        </div>
        <div>
            <div class="section-label" style="margin-bottom:0.25rem;">SEVERITY</div>
            <div style="font-family:'DM Mono',monospace; font-size:1.1rem; color:{sev_color};">{severity}</div>
        </div>
        <div style="flex:1; text-align:right;">
            <div style="font-size:0.72rem; color:#1de9b6; font-family:'DM Mono',monospace;">
                RAG-Augmented · Evidence-Based Clinical Report
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── RAG status chip ──────────────────────────────────────────────────────
    _render_rag_status()

    # ── Backend selector ─────────────────────────────────────────────────────
    backend = st.radio(
        "LLM Backend",
        options=[
            "📋 Instant Report (demo mode)",
            "⚡ Groq — Llama 3.1 8B (free key)",
            "✨ Gemini API (my key)",
        ],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    use_hardcoded = backend.startswith("📋")
    use_groq      = backend.startswith("⚡")
    use_gemini    = backend.startswith("✨")

    # ── RAG toggle (shown for all backends but cosmetic for hardcoded) ────────
    use_rag = st.checkbox(
        "🔍 Show RAG — display retrieved clinical knowledge base references",
        value=True,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # HARDCODED PATH (default)
    # ═════════════════════════════════════════════════════════════════════════
    if use_hardcoded:
        existing = st.session_state.get("gemini_explanation")
        col_btn, col_regen = st.columns([2, 1])
        with col_btn:
            generate = st.button(
                "📋 Generate Clinical Report",
                type="primary",
                use_container_width=True,
                key="gen_hardcoded",
            )
        with col_regen:
            regen = (
                st.button("🔄 Regenerate", use_container_width=True, key="regen_hardcoded")
                if existing else False
            )

        if generate or regen:
            if use_rag:
                try:
                    idx = get_rag_index()
                    if idx.is_ready:
                        q = build_query_from_results(
                            ensemble,
                            image_result   or {},
                            tabular_result or {},
                            st.session_state.get("feature_importance"),
                            (image_result or {}).get("modality_weights"),
                        )
                        _, rag_src = build_rag_context(q, idx)
                        st.session_state["rag_sources"] = rag_src
                except Exception:
                    pass

            report = get_hardcoded_report(kc_prob)
            st.session_state["gemini_explanation"] = report  # store dict

        if use_rag:
            rag_sources = st.session_state.get("rag_sources", [])
            if rag_sources:
                _render_rag_sources(rag_sources)

        report_data = st.session_state.get("gemini_explanation")
        if report_data:
            if isinstance(report_data, dict):
                _render_report_cards(report_data)
            else:
                _render_report_plain(report_data)

    # ═════════════════════════════════════════════════════════════════════════
    # GROQ PATH
    # ═════════════════════════════════════════════════════════════════════════
    elif use_groq:
        token = st.session_state.get("hf_token", "")
        if not token:
            st.markdown("""
            <div class="warning-banner">
                ⚡ Enter your <b>free</b> Groq API key in the sidebar.
            </div>
            """, unsafe_allow_html=True)
            _render_groq_help()
        else:
            _live_llm_section(
                ensemble, image_result, tabular_result,
                token, use_rag, backend_fn=generate_clinical_explanation_hf,
                btn_label="⚡ Generate (Groq · Llama 3.1 · Free)",
                spinner_msg="⚡ Llama 3.1 8B generating via Groq…",
                btn_key="gen_groq",
            )

    # ═════════════════════════════════════════════════════════════════════════
    # GEMINI PATH
    # ═════════════════════════════════════════════════════════════════════════
    elif use_gemini:
        token = st.session_state.get("gemini_api_key", "")
        if not token:
            st.markdown("""
            <div class="warning-banner">
                🔑 Enter your Gemini API key in the sidebar.
            </div>
            """, unsafe_allow_html=True)
        else:
            _live_llm_section(
                ensemble, image_result, tabular_result,
                token, use_rag, backend_fn=generate_gemini,
                btn_label="🧠 Generate (Gemini)",
                spinner_msg="🧠 Consulting Gemini…",
                btn_key="gen_gemini",
            )

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer" style="margin-top:1.5rem;">
        <b>⚠️ Important Disclaimer</b><br>
        This AI-generated report is for research and educational purposes only.
        It is <b>not</b> a medical diagnosis. All findings must be reviewed by
        a qualified ophthalmologist before any clinical decisions are made.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Live LLM helper (shared for Groq + Gemini)
# ─────────────────────────────────────────────────────────────────────────────

def _live_llm_section(ensemble, image_result, tabular_result,
                      token, use_rag, backend_fn,
                      btn_label, spinner_msg, btn_key):
    existing = st.session_state.get("gemini_explanation")
    col_btn, col_regen = st.columns([2, 1])
    with col_btn:
        generate = st.button(btn_label, type="primary",
                             use_container_width=True, key=btn_key)
    with col_regen:
        regen = (
            st.button("🔄 Regenerate", use_container_width=True,
                      key=f"regen_{btn_key}")
            if existing else False
        )

    if generate or regen:
        feature_importance = st.session_state.get("feature_importance")
        modality_weights   = (image_result or {}).get("modality_weights")

        rag_ctx, rag_src = "", []
        if use_rag:
            with st.spinner("📚 Retrieving clinical knowledge…"):
                try:
                    idx = get_rag_index()
                    if idx.is_ready:
                        q = build_query_from_results(
                            ensemble, image_result or {}, tabular_result or {},
                            feature_importance, modality_weights,
                        )
                        rag_ctx, rag_src = build_rag_context(q, idx)
                        st.session_state["rag_sources"] = rag_src
                except Exception as e:
                    st.warning(f"⚠️ RAG failed: {e}")

        with st.spinner(spinner_msg):
            explanation = backend_fn(
                hf_token=token,
                ensemble_result=ensemble,
                image_result=image_result   or {},
                tabular_result=tabular_result or {},
                feature_importance=feature_importance,
                modality_weights=modality_weights,
                rag_context=rag_ctx,
            )
            if explanation:
                st.session_state["gemini_explanation"] = explanation

    if use_rag:
        rag_sources = st.session_state.get("rag_sources", [])
        if rag_sources:
            _render_rag_sources(rag_sources)

    explanation = st.session_state.get("gemini_explanation")
    if explanation:
        _render_report(explanation)


# ─────────────────────────────────────────────────────────────────────────────
# Report renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_report_cards(report: dict):
    """Beautiful card-based renderer for the structured dict reports."""
    badge_color = report.get("badge_color", "#1de9b6")
    badge       = report.get("badge", "")
    icon        = report.get("icon", "🧠")

    # Header badge
    st.markdown(f"""
    <div style="
        margin: 1.2rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    ">
        <div style="
            font-size: 1.6rem;
            line-height: 1;
        ">{icon}</div>
        <div style="
            font-family: 'DM Serif Display', 'Georgia', serif;
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: {badge_color};
            text-transform: uppercase;
            border-left: 3px solid {badge_color};
            padding-left: 0.75rem;
        ">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

    # One card per section
    for i, section in enumerate(report.get("sections", [])):
        accent = ["#00e5ff", "#1de9b6", "#7c83fd"][i % 3]
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #0d1a2e 0%, #0a1525 100%);
            border: 1px solid #1e2d45;
            border-left: 3px solid {accent};
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.25);
        ">
            <div style="
                font-family: 'Inter', 'Segoe UI', sans-serif;
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: {accent};
                margin-bottom: 0.6rem;
            ">{section['title']}</div>
            <div style="
                font-family: 'Georgia', 'DM Serif Display', serif;
                font-size: 0.92rem;
                line-height: 1.85;
                color: #c8d6e8;
                font-weight: 400;
            ">{section['body']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer chip
    st.markdown(f"""
    <div style="
        background: rgba(255, 152, 0, 0.06);
        border: 1px solid rgba(255, 152, 0, 0.25);
        border-radius: 8px;
        padding: 0.65rem 1rem;
        margin-top: 0.5rem;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    ">
        <span style="font-size:0.9rem; flex-shrink:0; margin-top:1px;">⚠️</span>
        <span style="
            font-family: 'Inter', sans-serif;
            font-size: 0.76rem;
            color: #b0a070;
            line-height: 1.55;
        ">{report.get('disclaimer', '')}</span>
    </div>
    """, unsafe_allow_html=True)

    # Download button — build plain text version
    plain_text = f"# {badge}\n\n"
    for s in report.get("sections", []):
        import re
        body_plain = re.sub(r"<[^>]+>", "", s["body"])
        plain_text += f"## {s['title']}\n\n{body_plain}\n\n"
    plain_text += f"---\n\n{report.get('disclaimer', '')}"

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Report (.md)",
        data=plain_text,
        file_name="keratoscan_clinical_report.md",
        mime="text/markdown",
    )


def _render_report_plain(explanation: str):
    """Fallback renderer for plain-text reports (live LLM output)."""
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 Clinical Decision Support Report</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kc-card-accent" style="padding:1.75rem; line-height:1.85;
                                       color:#d0dbe8; font-size:0.9rem;
                                       font-family:'Georgia',serif;">
        {_md_to_html(explanation)}
    </div>
    """, unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Report (.md)",
        data=explanation,
        file_name="keratoscan_clinical_report.md",
        mime="text/markdown",
    )

# keep old name as alias for live LLM paths
def _render_report(explanation):
    if isinstance(explanation, dict):
        _render_report_cards(explanation)
    else:
        _render_report_plain(explanation)


def _render_rag_status():
    try:
        index = get_rag_index()
        stats = get_index_stats(index)
    except Exception:
        stats = {"ready": False, "num_documents": 0, "num_chunks": 0}

    if stats["ready"]:
        color = "#1de9b6"
        icon  = "✅"
        label = (f"RAG Active — {stats['num_documents']} document(s) · "
                 f"{stats['num_chunks']} chunks indexed")
    else:
        color = "#ff9800"
        icon  = "⚠️"
        label = "RAG Unavailable — add files to 'RAG Files/' folder"

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.6rem; background:#0d1320;
                border:1px solid #1e2d45; border-radius:8px;
                padding:0.55rem 1rem; margin:0.75rem 0; font-size:0.8rem;">
        <span>{icon}</span>
        <span style="color:{color}; font-family:'DM Mono',monospace;">{label}</span>
    </div>
    """, unsafe_allow_html=True)


def _render_rag_sources(sources: list):
    with st.expander(
        f"📚 Retrieved Knowledge Base References ({len(sources)} passages)",
        expanded=False,
    ):
        for src in sources:
            pct = int(src["score"] * 100)
            st.markdown(f"""
            <div style="background:#0d1320; border:1px solid #1e2d45;
                        border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.6rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                    <span style="font-family:'DM Mono',monospace; color:#00e5ff; font-size:0.8rem;">
                        [{src['index']}] {src['source']}
                    </span>
                    <span style="font-size:0.72rem; color:#8896b0;">
                        Relevance: <span style="color:#1de9b6;">{pct}%</span>
                    </span>
                </div>
                <div style="background:#1e2d45; height:3px; border-radius:2px; margin-bottom:0.5rem;">
                    <div style="width:{max(4, pct)}%; height:3px; border-radius:2px;
                                background:linear-gradient(90deg,#00e5ff,#1de9b6);"></div>
                </div>
                <div style="color:#8896b0; font-size:0.78rem; line-height:1.5;">
                    {src['preview']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def _render_groq_help():
    st.markdown("""
    <div class="kc-card" style="margin-top:1rem;">
        <div class="section-label">How to get a free Groq API key</div>
        <ol style="color:#8896b0; font-size:0.85rem; line-height:2.2;">
            <li>Go to <a href="https://console.groq.com" target="_blank"
                style="color:#00e5ff;">console.groq.com</a> → Sign Up (free)</li>
            <li>Click <b>API Keys</b> → <b>Create API Key</b></li>
            <li>Copy the key (starts with <code style="color:#1de9b6;">gsk_…</code>)</li>
            <li>Paste into the <b>Groq API Key</b> field in the sidebar</li>
        </ol>
        <div style="color:#3a4a60; font-size:0.78rem; margin-top:0.5rem;">
            ✅ Free · 14,400 req/day · No credit card · Llama 3.1 8B
        </div>
    </div>
    """, unsafe_allow_html=True)


def _md_to_html(text: str) -> str:
    import re
    # Tables → keep as-is wrapped in a scroll div
    text = re.sub(
        r"(\|.+\|\n)+",
        lambda m: f"<div style='overflow-x:auto;margin:0.75rem 0;'><table style='border-collapse:collapse;width:100%;font-size:0.82rem;'>"
                  + _table_to_html(m.group()) + "</table></div>",
        text,
    )
    text = re.sub(r"^## (.+)$",
                  r"<h3 style='color:#00e5ff;font-family:DM Serif Display,serif;"
                  r"margin:1.25rem 0 0.4rem;border-bottom:1px solid #1e2d45;padding-bottom:0.3rem;'>\1</h3>",
                  text, flags=re.MULTILINE)
    text = re.sub(r"^### (.+)$",
                  r"<h4 style='color:#1de9b6;margin:0.75rem 0 0.3rem;'>\1</h4>",
                  text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$",
                  r"<h2 style='color:#00e5ff;font-family:DM Serif Display,serif;'>\1</h2>",
                  text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#e8edf5;'>\1</b>", text)
    text = re.sub(r"^[\-\*] (.+)$",
                  r"<li style='margin:0.3rem 0;'>\1</li>",
                  text, flags=re.MULTILINE)
    text = re.sub(
        r"(<li.*?</li>\n?)+",
        lambda m: f"<ul style='margin:0.5rem 0 0.5rem 1rem;'>{m.group()}</ul>",
        text, flags=re.DOTALL,
    )
    text = re.sub(r"---+", "<hr style='border:none;border-top:1px solid #1e2d45;margin:1rem 0;'>", text)
    text = re.sub(r"\n\n", "<br>", text)
    text = re.sub(r"\n",   " ",    text)
    return text


def _table_to_html(md_table: str) -> str:
    """Convert a markdown table string to HTML rows."""
    rows = [r.strip() for r in md_table.strip().splitlines() if r.strip()]
    html = ""
    for i, row in enumerate(rows):
        if set(row.replace("|", "").replace("-", "").replace(" ", "")) == set():
            continue  # skip separator row
        cells = [c.strip() for c in row.split("|") if c.strip()]
        tag = "th" if i == 0 else "td"
        style = (
            "background:#0d1320;color:#00e5ff;padding:0.4rem 0.75rem;text-align:left;border-bottom:1px solid #1e2d45;"
            if i == 0
            else "color:#8896b0;padding:0.35rem 0.75rem;border-bottom:1px solid #111622;"
        )
        html += "<tr>" + "".join(f"<{tag} style='{style}'>{c}</{tag}>" for c in cells) + "</tr>"
    return html
