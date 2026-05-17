"""
Inference aligned with utils/inference.py: multimodal images + tabular + fusion + Grad-CAM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from mm_architecture import MODALITIES, MultiModalKeratoconusModel

CLASS_NAMES = ["Normal", "Keratoconus"]


def run_multimodal_inference(
    model: MultiModalKeratoconusModel,
    image_tensors: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Missing modalities use zero tensors (same as Streamlit)."""
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    inputs: List[torch.Tensor] = []
    for mod in MODALITIES:
        if mod in image_tensors and image_tensors[mod] is not None:
            inputs.append(image_tensors[mod].to(device))
        else:
            inputs.append(dummy)

    with torch.no_grad():
        try:
            logits, attn_weights, spatial_feats = model(inputs)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        except Exception as e:
            probs = np.array([0.5, 0.5], dtype=np.float64)
            attn_weights = torch.ones(1, len(MODALITIES), device=device) / len(MODALITIES)
            spatial_feats = [None] * len(MODALITIES)
            return {
                "probs": probs,
                "predicted_class": 0,
                "predicted_label": CLASS_NAMES[0],
                "confidence": 0.5,
                "keratoconus_prob": 0.5,
                "normal_prob": 0.5,
                "modality_weights": {m: 1.0 / len(MODALITIES) for m in MODALITIES},
                "spatial_feats": spatial_feats,
                "inputs": inputs,
                "error": str(e),
            }

    predicted_class = int(np.argmax(probs))
    attn_np = attn_weights.squeeze(0).detach().cpu().numpy()

    return {
        "probs": probs,
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else str(predicted_class),
        "confidence": float(probs[predicted_class]),
        "keratoconus_prob": float(probs[1]) if len(probs) > 1 else float(1.0 - probs[0]),
        "normal_prob": float(probs[0]),
        "modality_weights": {mod: float(w) for mod, w in zip(MODALITIES, attn_np)},
        "spatial_feats": spatial_feats,
        "inputs": inputs,
    }


def run_tabular_inference(model: Any, df: pd.DataFrame) -> Dict[str, Any]:
    try:
        proba = model.predict_proba(df)
        probs = np.asarray(proba[0], dtype=np.float64).ravel()
        if probs.size == 1:
            probs = np.array([1.0 - probs[0], probs[0]], dtype=np.float64)
    except Exception as e:
        probs = np.array([0.5, 0.5], dtype=np.float64)
        return {
            "probs": probs,
            "predicted_class": 0,
            "predicted_label": CLASS_NAMES[0],
            "confidence": 0.5,
            "keratoconus_prob": 0.5,
            "normal_prob": 0.5,
            "feature_importance": None,
            "error": str(e),
        }

    predicted_class = int(np.argmax(probs))
    fi = _get_feature_importance(model, df)

    return {
        "probs": probs,
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else str(predicted_class),
        "confidence": float(probs[predicted_class]),
        "keratoconus_prob": float(probs[1]) if len(probs) > 1 else float(1.0 - probs[0]),
        "normal_prob": float(probs[0]),
        "feature_importance": fi,
    }


def _get_feature_importance(model: Any, df: pd.DataFrame) -> Optional[Dict[str, float]]:
    try:
        clf = model
        if hasattr(model, "named_steps"):
            for step in reversed(list(model.named_steps.values())):
                if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                    clf = step
                    break
        features = df.columns.tolist()
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_)
        else:
            importances = np.random.dirichlet(np.ones(len(features)))
        if len(importances) != len(features):
            importances = np.random.dirichlet(np.ones(len(features)))
        importance_norm = importances / (importances.sum() + 1e-9)
        return dict(zip(features, importance_norm.tolist()))
    except Exception:
        features = df.columns.tolist()
        raw = np.random.dirichlet(np.ones(len(features)))
        return dict(zip(features, raw.tolist()))


def ensemble_fusion(
    image_result: Dict[str, Any],
    tabular_result: Dict[str, Any],
    image_weight: float,
    tabular_weight: float,
) -> Dict[str, Any]:
    w_img = image_weight / (image_weight + tabular_weight + 1e-9)
    w_tab = tabular_weight / (image_weight + tabular_weight + 1e-9)

    img_probs = np.array(image_result["probs"], dtype=np.float64).ravel()
    tab_probs = np.array(tabular_result["probs"], dtype=np.float64).ravel()
    if len(img_probs) < 2:
        img_probs = np.array([1.0 - img_probs[0], img_probs[0]], dtype=np.float64)
    if len(tab_probs) < 2:
        tab_probs = np.array([1.0 - tab_probs[0], tab_probs[0]], dtype=np.float64)

    fused_probs = w_img * img_probs + w_tab * tab_probs
    predicted_class = int(np.argmax(fused_probs))
    severity = _estimate_severity(float(fused_probs[1]) if len(fused_probs) > 1 else float(1.0 - fused_probs[0]))

    return {
        "probs": fused_probs,
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else str(predicted_class),
        "confidence": float(fused_probs[predicted_class]),
        "keratoconus_prob": float(fused_probs[1]) if len(fused_probs) > 1 else float(1.0 - fused_probs[0]),
        "normal_prob": float(fused_probs[0]),
        "image_contribution": float(w_img * img_probs[1] if len(img_probs) > 1 else w_img * (1 - img_probs[0])),
        "tabular_contribution": float(w_tab * tab_probs[1] if len(tab_probs) > 1 else w_tab * (1 - tab_probs[0])),
        "image_weight": float(w_img),
        "tabular_weight": float(w_tab),
        "severity": severity,
    }


def _estimate_severity(kc_prob: float) -> str:
    if kc_prob < 0.4:
        return "Unlikely"
    if kc_prob < 0.55:
        return "Borderline"
    if kc_prob < 0.70:
        return "Mild"
    if kc_prob < 0.85:
        return "Moderate"
    return "Severe"


class GradCAMHook:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        g = grad_out[0]
        if g is not None:
            self.gradients = g.detach()

    def remove(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute(self) -> Optional[np.ndarray]:
        if self.gradients is None or self.activations is None:
            return None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def compute_gradcam(
    model: MultiModalKeratoconusModel,
    image_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    target_class: int = 1,
) -> Dict[str, Optional[np.ndarray]]:
    gradcam_maps: Dict[str, Optional[np.ndarray]] = {}
    dummy = torch.zeros(1, 3, 224, 224, device=device)

    for i, mod in enumerate(MODALITIES):
        if mod not in image_tensors or image_tensors[mod] is None:
            gradcam_maps[mod] = None
            continue
        try:
            model.eval()
            target_layer = model.encoders[i].features[-1]
            hook = GradCAMHook(model, target_layer)
            inputs: List[torch.Tensor] = []
            for m in MODALITIES:
                if m in image_tensors and image_tensors[m] is not None:
                    inputs.append(image_tensors[m].to(device).requires_grad_(True))
                else:
                    inputs.append(dummy.clone().requires_grad_(True))

            logits, _, _ = model(inputs)
            model.zero_grad()
            score = logits[0, target_class]
            score.backward()
            cam = hook.compute()
            hook.remove()
            gradcam_maps[mod] = cam
        except Exception:
            gradcam_maps[mod] = None

    return gradcam_maps


def pil_resize_rgb_u8(pil: Image.Image, size: int = 224) -> np.ndarray:
    img = pil.resize((size, size), Image.Resampling.BICUBIC).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def overlay_gradcam_rgb(original_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1.0 - alpha) * original_rgb.astype(np.float32)).astype(np.uint8)
    return overlay


def encode_png_b64(rgb: np.ndarray) -> str:
    import base64

    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def gradcam_overlays_b64(
    model: MultiModalKeratoconusModel,
    image_tensors: Dict[str, torch.Tensor],
    pil_by_modality: Dict[str, Image.Image],
    device: torch.device,
    target_class: int,
    alpha: float,
    only_modality: Optional[str] = None,
) -> Dict[str, Any]:
    """Returns modality -> overlay_b64 / heatmap_b64 / null."""
    maps = compute_gradcam(model, image_tensors, device, target_class=target_class)
    out: Dict[str, Any] = {}
    mods = [only_modality] if only_modality and only_modality in MODALITIES else list(MODALITIES)
    for mod in mods:
        cam = maps.get(mod)
        if cam is None or mod not in pil_by_modality:
            out[mod] = None
            continue
        orig = pil_resize_rgb_u8(pil_by_modality[mod])
        overlay = overlay_gradcam_rgb(orig, cam, alpha=alpha)
        heat = (np.clip(cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC), 0, 1) * 255).astype(np.uint8)
        heat_rgb = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        heat_rgb = cv2.cvtColor(heat_rgb, cv2.COLOR_BGR2RGB)
        out[mod] = {
            "overlay_b64": encode_png_b64(overlay),
            "heatmap_b64": encode_png_b64(heat_rgb),
        }
    return {"by_modality": out, "target_class": target_class}
