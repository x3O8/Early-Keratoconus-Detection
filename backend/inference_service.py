from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import clinical_text
import fusion
import mm_inference
import model_service
from mm_architecture import MODALITIES
from preprocessor import preprocess_tensor, pil_from_bytes, tensor_to_display_u8, validate_csv


def _default_image_result() -> Dict[str, Any]:
    return {
        "probs": np.array([0.5, 0.5], dtype=np.float64),
        "predicted_class": 0,
        "predicted_label": "Normal",
        "confidence": 0.5,
        "keratoconus_prob": 0.5,
        "normal_prob": 0.5,
        "modality_weights": {m: 1.0 / len(MODALITIES) for m in MODALITIES},
        "spatial_feats": [],
        "inputs": [],
    }


def _default_tabular_result() -> Dict[str, Any]:
    return {
        "probs": np.array([0.5, 0.5], dtype=np.float64),
        "predicted_class": 0,
        "predicted_label": "Normal",
        "confidence": 0.5,
        "keratoconus_prob": 0.5,
        "normal_prob": 0.5,
        "feature_importance": {},
    }


def parse_modalities_from_request(request) -> Tuple[Dict[str, torch.Tensor], Dict[str, Image.Image], List[str]]:
    tensors: Dict[str, torch.Tensor] = {}
    pils: Dict[str, Image.Image] = {}
    warnings: List[str] = []
    for m in MODALITIES:
        if m not in request.files:
            continue
        f = request.files[m]
        if not f or not getattr(f, "filename", None):
            continue
        raw = f.read()
        if not raw:
            continue
        pil = pil_from_bytes(raw)
        tensors[m] = preprocess_tensor(pil)
        pils[m] = pil
    missing = [m for m in MODALITIES if m not in tensors]
    if missing:
        warnings.append(f"Missing modalities (zero-filled): {', '.join(missing)}")
    return tensors, pils, warnings


def run_full_analysis(
    image_tensors: Dict[str, torch.Tensor],
    df_tabular: Optional[pd.DataFrame],
    w_image: float,
    w_tabular: float,
    cache: model_service.AppModelCache,
) -> Dict[str, Any]:
    device = cache.device
    has_images = len(image_tensors) > 0
    has_tabular = df_tabular is not None

    if not has_images and not has_tabular:
        raise ValueError("Provide at least one modality image and/or a valid tabular CSV.")

    image_result: Optional[Dict[str, Any]] = None
    if cache.image_model is not None and has_images:
        image_result = mm_inference.run_multimodal_inference(cache.image_model, image_tensors, device)
    elif cache.image_model is not None:
        image_result = mm_inference.run_multimodal_inference(cache.image_model, {}, device)

    tabular_result: Optional[Dict[str, Any]] = None
    if cache.tabular_model is not None and has_tabular:
        tabular_result = mm_inference.run_tabular_inference(cache.tabular_model, df_tabular)

    if image_result is None:
        image_result = _default_image_result()
    if tabular_result is None:
        tabular_result = _default_tabular_result()

    ensemble = mm_inference.ensemble_fusion(image_result, tabular_result, w_image, w_tabular)
    ensemble_json = {
        **ensemble,
        "probs": np.asarray(ensemble["probs"], dtype=np.float64).ravel().tolist(),
    }

    p_img = np.array(image_result["probs"], dtype=np.float64).ravel()
    p_tab = np.array(tabular_result["probs"], dtype=np.float64).ravel()
    p_fused = np.array(ensemble["probs"], dtype=np.float64).ravel()

    ent = fusion.entropy(p_fused)
    max_ent = float(np.log(max(len(p_fused), 2)))
    uncertainty = float(ent / max_ent) if max_ent > 0 else 0.0
    agreement = fusion.model_agreement(p_img, p_tab)

    narrative = clinical_text.build_clinical_narrative_multimodal(
        fused_probs=p_fused,
        image_probs=p_img,
        tabular_probs=p_tab,
        agreement=agreement,
        uncertainty=uncertainty,
        w_image=w_image,
        w_tabular=w_tabular,
        modality_weights=image_result.get("modality_weights") or {},
        ensemble_severity=ensemble.get("severity", ""),
    )

    previews: Dict[str, str] = {}
    for mod, t in image_tensors.items():
        arr = tensor_to_display_u8(t.squeeze(0))
        ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if ok:
            previews[mod] = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "modalities_received": list(image_tensors.keys()),
        "n_modalities_uploaded": len(image_tensors),
        "tabular_loaded": has_tabular,
        "image_model": {
            "probs": p_img.tolist(),
            "predicted_label": image_result.get("predicted_label"),
            "confidence": float(image_result.get("confidence", 0)),
            "modality_weights": {k: float(v) for k, v in (image_result.get("modality_weights") or {}).items()},
            "error": image_result.get("error"),
        },
        "tabular_model": {
            "probs": p_tab.tolist(),
            "predicted_label": tabular_result.get("predicted_label"),
            "confidence": float(tabular_result.get("confidence", 0)),
            "feature_importance": {
                str(k): float(v) for k, v in (tabular_result.get("feature_importance") or {}).items()
            },
            "error": tabular_result.get("error"),
        },
        "ensemble": ensemble_json,
        "fusion": {"w_image": float(w_image), "w_tabular": float(w_tabular)},
        "metrics": {
            "fused_probs": p_fused.tolist(),
            "predicted_class": int(np.argmax(p_fused)),
            "predicted_label": ensemble.get("predicted_label", ""),
            "confidence": float(ensemble.get("confidence", 0)),
            "uncertainty_normalized": uncertainty,
            "entropy": ent,
            "model_agreement": agreement,
            "keratoconus_risk": float(ensemble.get("keratoconus_prob", 0)),
            "normal_prob": float(ensemble.get("normal_prob", 0)),
            "image_contribution_kc": float(ensemble.get("image_contribution", 0)),
            "tabular_contribution_kc": float(ensemble.get("tabular_contribution", 0)),
            "severity": ensemble.get("severity", ""),
            "clinical": narrative,
        },
        "previews": previews,
    }
