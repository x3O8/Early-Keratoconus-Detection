"""
Gemini LLM Integration — RAG-augmented clinical explanation generation.
"""

import streamlit as st
from typing import Optional, Dict


def generate_clinical_explanation(
    api_key: str,
    ensemble_result: Dict,
    image_result: Dict,
    tabular_result: Dict,
    feature_importance: Optional[Dict] = None,
    modality_weights: Optional[Dict] = None,
    rag_context: Optional[str] = None,
) -> Optional[str]:
    """
    Call Gemini API to generate a clinical explanation based on model outputs.
    Optionally enriched with RAG-retrieved medical literature context.

    Returns the explanation string or None on failure.
    """
    if not api_key or not api_key.strip():
        st.error("Please enter a valid Gemini API key in the sidebar.")
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        st.error("google-generativeai package not installed. Run: pip install google-generativeai")
        return None

    # ── Build rich context from model outputs ────────────────────────────────
    kc_prob   = ensemble_result.get("keratoconus_prob", 0.0)
    severity  = ensemble_result.get("severity", "Unknown")
    diagnosis = ensemble_result.get("predicted_label", "Unknown")
    confidence = ensemble_result.get("confidence", 0.0)

    img_kc     = image_result.get("keratoconus_prob", 0.0)
    tab_kc     = tabular_result.get("keratoconus_prob", 0.0)
    img_weight = ensemble_result.get("image_weight", 0.65)
    tab_weight = ensemble_result.get("tabular_weight", 0.35)

    # Top modalities
    mod_summary = ""
    if modality_weights:
        top_mods = sorted(modality_weights.items(), key=lambda x: -x[1])[:3]
        mod_summary = ", ".join([f"{m} ({w:.1%})" for m, w in top_mods])

    # Top features
    feat_summary = ""
    if feature_importance:
        top_feats = sorted(feature_importance.items(), key=lambda x: -x[1])[:5]
        feat_summary = ", ".join([f"{f} ({v:.1%})" for f, v in top_feats])

    # ── Build RAG section ────────────────────────────────────────────────────
    rag_section = ""
    if rag_context and rag_context.strip():
        rag_section = f"""
## Evidence Base (Retrieved from Clinical Knowledge Base)

{rag_context}

---

Using the above retrieved clinical evidence as factual grounding, now generate the report below.
Reference specific passages where relevant. Do NOT hallucinate thresholds or values not in the evidence.

"""

    # ── Assemble prompt ──────────────────────────────────────────────────────
    prompt = f"""You are a specialist ophthalmic AI assistant providing clinical decision support.
You have access to a curated clinical knowledge base (provided below as retrieved references).

## AI System Analysis Summary

**Ensemble Diagnosis:** {diagnosis}
**Keratoconus Probability:** {kc_prob:.1%}
**Estimated Severity:** {severity}
**System Confidence:** {confidence:.1%}

**Model Breakdown:**
- Multi-modal Image Model (weight {img_weight:.0%}): {img_kc:.1%} keratoconus probability
- Tabular Clinical Model (weight {tab_weight:.0%}): {tab_kc:.1%} keratoconus probability

**Key Contributing Image Modalities:** {mod_summary if mod_summary else "Not available"}
**Top Discriminative Clinical Features:** {feat_summary if feat_summary else "Not available"}

---
{rag_section}
Please provide a structured clinical decision support report with the following sections:

## 1. Clinical Interpretation
Explain what the AI findings suggest clinically, referencing the specific modalities and features
that drove the prediction. Be specific about what elevation maps, curvature maps, or clinical
indices indicate. Ground your interpretation in the retrieved clinical evidence where applicable.

## 2. Prediction Reasoning
Describe the reasoning behind the prediction, how the two model streams agreed or disagreed,
and what the confidence level implies clinically.

## 3. Evidence-Based Context
Briefly cite what the retrieved clinical literature says about this severity level, relevant
thresholds (Kmax, pachymetry, elevation values), and how this case fits within established
classification systems (Amsler-Krumeich, ABCD, BAD-D).

## 4. Suggested Management
Based on the severity estimate and retrieved protocols, suggest appropriate next steps:
- Observation and repeat topography intervals
- Corneal cross-linking (CXL) evaluation criteria and urgency
- Contact lens fitting strategy (RGP, scleral, hybrid)
- Referral thresholds and surgical candidacy
- Contraindications (e.g., LASIK in KC patients)

## 5. Monitoring Recommendations
Specify follow-up intervals and metrics to monitor (Kmax, thinnest point, BSCVA, posterior elevation, etc.)

## 6. Patient-Friendly Summary
Write 2-3 sentences a non-specialist patient can understand explaining the finding and next step.

## Important Caveats
Close with a clear statement that this is an AI-assisted analysis, not a clinical diagnosis,
and must be reviewed by a qualified ophthalmologist before any clinical decision is made.

Keep the tone professional, medically accurate, and appropriately cautious.
Do not hallucinate specific numerical values not provided in the AI summary or the retrieved references."""

    try:
        genai.configure(api_key=api_key.strip())
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
            ),
        )
        return response.text
    except Exception as e:
        err_msg = str(e)
        if "API_KEY" in err_msg.upper() or "invalid" in err_msg.lower():
            st.error("❌ Invalid Gemini API key. Please check your key in the sidebar.")
        elif "quota" in err_msg.lower():
            st.error("❌ Gemini API quota exceeded. Please try again later.")
        else:
            st.error(f"❌ Gemini API error: {err_msg}")
        return None
