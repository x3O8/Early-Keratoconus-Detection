"""
Free LLM client for KeratoScan AI — Groq API (Llama 3.1 8B).

Groq provides completely free access to open-source LLMs including Llama 3.1.
Free tier: 14,400 requests/day, 6,000 tokens/min. No billing required.

Get a free API key at: https://console.groq.com
  → Sign Up (free) → API Keys → Create API Key
"""

import streamlit as st
from typing import Optional, Dict

GROQ_MODEL     = "llama-3.1-8b-instant"
MAX_TOKENS     = 2000
TEMPERATURE    = 0.3


def generate_clinical_explanation_hf(
    hf_token: str,
    ensemble_result: Dict,
    image_result: Dict,
    tabular_result: Dict,
    feature_importance: Optional[Dict] = None,
    modality_weights: Optional[Dict] = None,
    rag_context: Optional[str] = None,
) -> Optional[str]:
    """
    Generate a RAG-augmented clinical explanation via Groq API (Llama 3.1 8B).

    The parameter is named hf_token for interface compatibility, but it expects
    a Groq API key from console.groq.com (free account).
    """
    api_key = hf_token
    if not api_key or not api_key.strip():
        st.error("Please enter your Groq API key in the sidebar.")
        return None

    try:
        from groq import Groq
    except ImportError:
        st.error("groq package not installed. Run: pip install groq")
        return None

    # ── Build context ─────────────────────────────────────────────────────────
    kc_prob    = ensemble_result.get("keratoconus_prob", 0.0)
    severity   = ensemble_result.get("severity", "Unknown")
    diagnosis  = ensemble_result.get("predicted_label", "Unknown")
    confidence = ensemble_result.get("confidence", 0.0)
    img_kc     = image_result.get("keratoconus_prob", 0.0)
    tab_kc     = tabular_result.get("keratoconus_prob", 0.0)
    img_w      = ensemble_result.get("image_weight", 0.65)
    tab_w      = ensemble_result.get("tabular_weight", 0.35)

    mod_summary = ""
    if modality_weights:
        top = sorted(modality_weights.items(), key=lambda x: -x[1])[:3]
        mod_summary = ", ".join(f"{m} ({w:.1%})" for m, w in top)

    feat_summary = ""
    if feature_importance:
        top = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
        feat_summary = ", ".join(f"{f} ({v:.1%})" for f, v in top)

    rag_section = ""
    if rag_context and rag_context.strip():
        rag_section = (
            "\n\n## Retrieved Clinical Evidence\n\n"
            + rag_context
            + "\n\nGround your report in the above evidence. "
              "Do NOT invent thresholds or values not stated.\n"
        )

    system = (
        "You are a specialist ophthalmic AI assistant providing clinical decision support. "
        "Write structured, evidence-based reports. Be medically accurate, professional, "
        "and appropriately cautious. Always include a disclaimer that AI findings must be "
        "independently reviewed by a qualified ophthalmologist before any clinical decision."
    )

    user_msg = f"""## AI System Analysis Results

**Ensemble Diagnosis:** {diagnosis}
**Keratoconus Probability:** {kc_prob:.1%}
**Severity Estimate:** {severity}
**System Confidence:** {confidence:.1%}

**Model Breakdown:**
- Image Model (weight {img_w:.0%}): {img_kc:.1%} KC probability
- Tabular Clinical Model (weight {tab_w:.0%}): {tab_kc:.1%} KC probability

**Key Contributing Modalities:** {mod_summary or "Not available"}
**Top Discriminative Clinical Features:** {feat_summary or "Not available"}
{rag_section}
---

Write a structured clinical decision support report with these exact sections:

## 1. Clinical Interpretation
What do the AI findings suggest clinically? Reference specific modalities and features.

## 2. Prediction Reasoning
How did the two model streams agree or disagree? What does the confidence level imply?

## 3. Evidence-Based Context
What does clinical literature say about this severity level and relevant thresholds?

## 4. Suggested Management
Based on severity: observation intervals, CXL evaluation, contact lens strategy (RGP/scleral/hybrid), referral thresholds, surgical candidacy, contraindications.

## 5. Monitoring Recommendations
Follow-up frequency and key metrics to track (Kmax, thinnest pachymetry, BSCVA, posterior elevation).

## 6. Patient-Friendly Summary
2-3 plain-language sentences for a non-specialist patient.

## Important Caveats
State clearly this is AI-assisted analysis only and must be reviewed by an ophthalmologist."""

    # ── Call Groq API ─────────────────────────────────────────────────────────
    try:
        client = Groq(api_key=api_key.strip())

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        text = response.choices[0].message.content
        return text.strip() if text else None

    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower() or "authentication" in err.lower():
            st.error(
                "❌ Invalid Groq API key. "
                "Get a free key at console.groq.com → API Keys."
            )
        elif "429" in err or "rate_limit" in err.lower():
            st.error("❌ Groq rate limit hit. Please wait a moment and try again.")
        elif "connection" in err.lower() or "timeout" in err.lower():
            st.error(
                "❌ Cannot reach Groq API. Check your internet connection. "
                "If behind a corporate proxy, contact your IT admin to whitelist api.groq.com."
            )
        else:
            st.error(f"❌ Groq API error: {err}")
        return None
