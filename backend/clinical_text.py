from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def class_label_names(num_classes: int) -> List[str]:
    base = ["Normal", "Keratoconus"]
    if num_classes <= len(base):
        return base[:num_classes]
    return [f"Class {i}" for i in range(num_classes)]


def build_clinical_narrative_multimodal(
    fused_probs: np.ndarray,
    image_probs: np.ndarray,
    tabular_probs: np.ndarray,
    agreement: float,
    uncertainty: float,
    w_image: float,
    w_tabular: float,
    modality_weights: Dict[str, float],
    ensemble_severity: str,
) -> Dict[str, str]:
    names = class_label_names(len(fused_probs))
    pred = int(np.argmax(fused_probs))
    conf = float(fused_probs[pred])
    kc_idx = min(1, len(fused_probs) - 1)
    kc_risk = float(fused_probs[kc_idx]) if len(fused_probs) > 1 else float(1.0 - fused_probs[0])

    lines: List[str] = []

    if modality_weights:
        top = sorted(modality_weights.items(), key=lambda x: -x[1])[:3]
        top_s = ", ".join(f"{k} ({v*100:.0f}%)" for k, v in top)
        lines.append(f"Highest cross-modal attention on: {top_s}.")

    if kc_risk >= 0.65:
        lines.append(
            "Corneal asymmetry and inferior steepening patterns on the fused imaging stack suggest elevated early keratoconus risk."
        )
    elif kc_risk >= 0.45:
        lines.append(
            "Suspicious curvature signal across modalities without definitive collapse — correlate with topography and tomography."
        )
    else:
        lines.append("Fused imaging and clinical scores skew toward a more normal corneal risk profile on this assessment.")

    lines.append(
        f"Ensemble reads {names[pred]} at {conf*100:.1f}% confidence; normalized uncertainty {uncertainty:.3f}."
    )

    if agreement >= 0.85:
        lines.append("Imaging and tabular models largely agree, improving confidence in the fused estimate.")
    elif agreement >= 0.65:
        lines.append("Moderate model agreement — inspect per-modality Grad-CAM for discordant regions.")
    else:
        lines.append("High disagreement between imaging and tabular branches; prioritize clinical examination over the score.")

    iw = w_image / (w_image + w_tabular + 1e-9)
    tw = w_tabular / (w_image + w_tabular + 1e-9)
    lines.append(f"Fusion mix: imaging weight {iw*100:.0f}%, tabular weight {tw*100:.0f}%.")

    lines.append(f"Severity index (KC probability–based): {ensemble_severity}.")

    return {
        "summary": " ".join(lines),
        "bullets": lines,
        "severity_band": ensemble_severity or progression_band(kc_risk),
    }


def progression_band(kc_prob: float) -> str:
    if kc_prob < 0.35:
        return "Low"
    if kc_prob < 0.55:
        return "Borderline"
    if kc_prob < 0.72:
        return "Mild"
    if kc_prob < 0.86:
        return "Moderate"
    return "High"
