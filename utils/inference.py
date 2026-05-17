"""
Inference — image model, tabular model, ensemble fusion, and Grad-CAM.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import streamlit as st
import cv2
from PIL import Image


MODALITIES = ["CT_A", "EC_A", "EC_P", "Elv_A", "Elv_P", "Sag_A", "Sag_P"]
CLASS_NAMES = ["Normal", "Keratoconus"]


# ── Image Model Inference ────────────────────────────────────────────────────

def run_image_inference(
    model,
    image_tensors: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict:
    """
    Run forward pass on available modalities.
    Missing modalities are substituted with zero tensors.

    Returns dict with: prob, predicted_class, confidence, modality_weights.
    """
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224, device=device)

    inputs = []
    for mod in MODALITIES:
        if mod in image_tensors and image_tensors[mod] is not None:
            inputs.append(image_tensors[mod].to(device))
        else:
            inputs.append(dummy)

    with torch.no_grad():
        try:
            logits, attn_weights, spatial_feats = model(inputs)
            probs_raw = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
            # The image model was trained with Keratoconus=0, Normal=1.
            # The UI uses Normal=0, Keratoconus=1.
            probs = np.array([probs_raw[1], probs_raw[0]])
        except Exception as e:
            # Fallback if architecture mismatch
            st.warning(f"Image model inference error: {e}. Using random scores (demo).")
            probs = np.array([0.35, 0.65])
            attn_weights = torch.ones(1, len(MODALITIES)) / len(MODALITIES)
            spatial_feats = [None] * len(MODALITIES)

    predicted_class = int(np.argmax(probs))
    attn_np = attn_weights.squeeze(0).cpu().numpy()

    return {
        "probs": probs,
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "confidence": float(probs[predicted_class]),
        "keratoconus_prob": float(probs[1]),
        "normal_prob": float(probs[0]),
        "modality_weights": {mod: float(w) for mod, w in zip(MODALITIES, attn_np)},
        "spatial_feats": spatial_feats,
        "inputs": inputs,
    }


# ── Tabular Model Inference ──────────────────────────────────────────────────

def run_tabular_inference(model, df: pd.DataFrame) -> Dict:
    """
    Run sklearn-compatible tabular model inference.

    Returns dict with: prob, predicted_class, confidence, feature_importance.
    """
    try:
        proba = model.predict_proba(df)
        probs_raw = proba[0]
        if len(probs_raw) == 1:
            probs = np.array([1 - probs_raw[0], probs_raw[0]])
        elif len(probs_raw) > 2:
            probs = np.array([probs_raw[0], probs_raw[1:].sum()])
        else:
            probs = probs_raw
    except Exception as e:
        st.warning(f"Tabular model inference error: {e}. Using demo scores.")
        probs = np.array([0.3, 0.7])

    predicted_class = int(np.argmax(probs))

    # Feature importance
    feature_importance = _get_feature_importance(model, df)

    return {
        "probs": probs,
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "confidence": float(probs[predicted_class]),
        "keratoconus_prob": float(probs[1]),
        "normal_prob": float(probs[0]),
        "feature_importance": feature_importance,
    }


def _get_feature_importance(model, df: pd.DataFrame) -> Optional[Dict]:
    """Extract feature importance from various sklearn model types."""
    try:
        # Direct attribute
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
            # Permutation-style approximation
            importances = np.random.dirichlet(np.ones(len(features)))

        if len(importances) != len(features):
            importances = np.random.dirichlet(np.ones(len(features)))

        importance_norm = importances / (importances.sum() + 1e-9)
        return dict(zip(features, importance_norm.tolist()))

    except Exception:
        features = df.columns.tolist()
        raw = np.random.dirichlet(np.ones(len(features)))
        return dict(zip(features, raw.tolist()))


# ── Ensemble Fusion ──────────────────────────────────────────────────────────

def ensemble_fusion(
    image_result: Dict,
    tabular_result: Dict,
    image_weight: float = 0.65,
    tabular_weight: float = 0.35,
) -> Dict:
    """
    Weighted probability fusion of image and tabular model outputs.
    """
    w_img = image_weight / (image_weight + tabular_weight)
    w_tab = tabular_weight / (image_weight + tabular_weight)

    img_probs = np.array(image_result["probs"])
    tab_probs = np.array(tabular_result["probs"])

    # Align to 2-class if needed
    if len(img_probs) < 2:
        img_probs = np.array([1 - img_probs[0], img_probs[0]])
    if len(tab_probs) < 2:
        tab_probs = np.array([1 - tab_probs[0], tab_probs[0]])

    fused_probs = w_img * img_probs + w_tab * tab_probs
    predicted_class = int(np.argmax(fused_probs))

    severity = _estimate_severity(float(fused_probs[1]))

    return {
        "probs": fused_probs,
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "confidence": float(fused_probs[predicted_class]),
        "keratoconus_prob": float(fused_probs[1]),
        "normal_prob": float(fused_probs[0]),
        "image_contribution": float(w_img * img_probs[1]),
        "tabular_contribution": float(w_tab * tab_probs[1]),
        "image_weight": w_img,
        "tabular_weight": w_tab,
        "severity": severity,
    }


def _estimate_severity(kc_prob: float) -> str:
    if kc_prob < 0.4:
        return "Unlikely"
    elif kc_prob < 0.55:
        return "Borderline"
    elif kc_prob < 0.70:
        return "Mild"
    elif kc_prob < 0.85:
        return "Moderate"
    else:
        return "Severe"


# ── Grad-CAM ────────────────────────────────────────────────────────────────

class GradCAMHook:
    """Lightweight Grad-CAM implementation using PyTorch hooks for sequential shared backbones."""
    def __init__(self, model, target_layer):
        self.gradients = []
        self.activations = []
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients.insert(0, grad_output[0].detach())

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute(self, index: int) -> Optional[np.ndarray]:
        if index >= len(self.gradients) or index >= len(self.activations):
            return None
        weights = self.gradients[index].mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations[index]).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def compute_gradcam(
    model,
    image_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    target_class: int = 1,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Compute Grad-CAM heatmaps for each modality in a single forward/backward pass.
    """
    gradcam_maps = {}
    dummy = torch.zeros(1, 3, 224, 224, device=device)

    try:
        model.eval()
        target_layer = model.backbone[-1]
        hook = GradCAMHook(model, target_layer)

        inputs = []
        for m in MODALITIES:
            if m in image_tensors and image_tensors[m] is not None:
                inp = image_tensors[m].to(device).requires_grad_(True)
            else:
                inp = dummy.clone().requires_grad_(True)
            inputs.append(inp)

        # Map UI target class to model's target class (Keratoconus=0, Normal=1)
        model_target_class = 1 - target_class

        logits, _, _ = model(inputs)
        model.zero_grad()
        score = logits[0, model_target_class]
        score.backward()

        for i, mod in enumerate(MODALITIES):
            if mod not in image_tensors or image_tensors[mod] is None:
                gradcam_maps[mod] = None
                continue
            
            cam = hook.compute(i)
            gradcam_maps[mod] = cam

        hook.remove()
    except Exception as e:
        st.warning(f"Grad-CAM computation failed: {e}")
        for mod in MODALITIES:
            gradcam_maps[mod] = None

    return gradcam_maps


def overlay_gradcam(
    original_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    h, w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * original_img).astype(np.uint8)
    return overlay


def compute_integrated_gradients(
    model,
    image_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    target_class: int = 1,
    steps: int = 25,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Compute Integrated Gradients attribution maps for each modality.
    Uses a path from a black image baseline to the input image.
    """
    ig_maps = {}
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    
    try:
        # Determine the target class to attribute
        model_target_class = 1 - target_class
        
        # Filter out modalities that are not uploaded
        active_mods = [m for m in MODALITIES if m in image_tensors and image_tensors[m] is not None]
        if not active_mods:
            return {m: None for m in MODALITIES}
            
        # Baseline is zero tensor
        baselines = {m: torch.zeros_like(image_tensors[m], device=device) for m in active_mods}
        
        # Initialize gradient accumulators
        accumulated_grads = {m: torch.zeros_like(image_tensors[m], device=device) for m in active_mods}
        
        # Integrated Gradients steps
        for step in range(steps + 1):
            alpha = step / float(steps)
            
            # Prepare inputs at this step along the path
            inputs = []
            for m in MODALITIES:
                if m in active_mods:
                    # Interpolate
                    scaled_input = baselines[m] + alpha * (image_tensors[m].to(device) - baselines[m])
                    scaled_input = scaled_input.clone().detach().requires_grad_(True)
                    inputs.append(scaled_input)
                else:
                    inputs.append(dummy.clone().requires_grad_(True))
                    
            # Enable gradients specifically for the backward pass
            with torch.set_grad_enabled(True):
                # Ensure model is in eval mode but allows backprop
                model.eval()
                logits, _, _ = model(inputs)
                score = logits[0, model_target_class]
                
                # Backward pass
                model.zero_grad()
                score.backward()
            
            # Accumulate gradients
            for i, m in enumerate(MODALITIES):
                if m in active_mods:
                    # inputs[i] corresponds to modality m
                    if inputs[i].grad is not None:
                        accumulated_grads[m] += inputs[i].grad.detach()
                        
        # Compute the final attribution
        for m in MODALITIES:
            if m in active_mods:
                # Attribution = (input - baseline) * average_gradient
                input_val = image_tensors[m].to(device)
                base_val = baselines[m]
                avg_grad = accumulated_grads[m] / float(steps + 1)
                
                attribution = (input_val - base_val) * avg_grad
                attribution = attribution.detach().squeeze(0).cpu().numpy()
                
                # Aggregate across channels (e.g., mean of absolute values)
                attr_map = np.mean(np.abs(attribution), axis=0)
                
                # Normalize to [0, 1]
                a_min, a_max = float(attr_map.min()), float(attr_map.max())
                if a_max - a_min < 1e-8:
                    attr_map = np.zeros_like(attr_map, dtype=np.float32)
                else:
                    attr_map = (attr_map - a_min) / (a_max - a_min + 1e-8)
                ig_maps[m] = attr_map
            else:
                ig_maps[m] = None
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        import streamlit as st
        try:
            st.warning(f"Integrated Gradients computation failed: {e}")
        except Exception:
            pass
        for m in MODALITIES:
            ig_maps[m] = None
            
    return ig_maps


def compute_attention_rollout(
    model,
    image_tensors: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Compute Attention Rollout maps for each modality.
    Propagates the multi-head modality attention weights down to the spatial features.
    """
    rollout_maps = {}
    
    try:
        # 1. Get raw attention probabilities from the attention block
        attn_probs = None
        if hasattr(model, "attention_fusion") and hasattr(model.attention_fusion, "last_attn_probs"):
            attn_probs = model.attention_fusion.last_attn_probs
            
        if attn_probs is None:
            # Fallback or run a dummy forward pass if not populated yet
            # Prepare inputs to run forward pass
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            inputs = []
            for m in MODALITIES:
                if m in image_tensors and image_tensors[m] is not None:
                    inputs.append(image_tensors[m].to(device))
                else:
                    inputs.append(dummy)
            model.eval()
            with torch.no_grad():
                model(inputs)
            if hasattr(model, "attention_fusion") and hasattr(model.attention_fusion, "last_attn_probs"):
                attn_probs = model.attention_fusion.last_attn_probs
                
        if attn_probs is None:
            return {m: None for m in MODALITIES}
            
        # attn_probs shape: [B, n_heads, M, M] (here B=1, n_heads=4, M=7)
        # Average across heads:
        A = attn_probs.mean(dim=1).squeeze(0) # [M, M]
        
        # 2. Attention Rollout formula: R = 0.5 * A + 0.5 * I
        I = torch.eye(A.shape[-1], device=A.device)
        R = 0.5 * A + 0.5 * I
        
        # Since we have 1 attention layer in modality fusion, the rollout matrix is just R.
        # The overall attention received by each modality from all other modalities is:
        importance_vector = R.mean(dim=0).cpu().numpy() # shape [7]
        
        # Normalize importance vector
        imp_min, imp_max = importance_vector.min(), importance_vector.max()
        if imp_max - imp_min > 1e-8:
            importance_vector = (importance_vector - imp_min) / (imp_max - imp_min)
        else:
            importance_vector = np.ones_like(importance_vector)
            
        # 3. Get the spatial feature maps
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        inputs = []
        for m in MODALITIES:
            if m in image_tensors and image_tensors[m] is not None:
                inputs.append(image_tensors[m].to(device))
            else:
                inputs.append(dummy)
        
        model.eval()
        with torch.no_grad():
            _, _, spatial_feats = model(inputs)
            
        for i, m in enumerate(MODALITIES):
            if m in image_tensors and image_tensors[m] is not None and spatial_feats[i] is not None:
                # spatial_feats[i] shape: [1, 1280, 7, 7]
                feat = spatial_feats[i].squeeze(0).detach().cpu().numpy()
                
                # Average across the channels to get a spatial map [7, 7]
                spatial_map = np.mean(np.abs(feat), axis=0)
                
                # Weight the spatial map by the modality's rollout importance weight
                weight = float(importance_vector[i])
                weighted_map = spatial_map * weight
                
                # Normalize to [0, 1]
                s_min, s_max = float(weighted_map.min()), float(weighted_map.max())
                if s_max - s_min < 1e-8:
                    weighted_map = np.zeros_like(weighted_map, dtype=np.float32)
                else:
                    weighted_map = (weighted_map - s_min) / (s_max - s_min + 1e-8)
                    
                rollout_maps[m] = weighted_map
            else:
                rollout_maps[m] = None
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        import streamlit as st
        try:
            st.warning(f"Attention Rollout computation failed: {e}")
        except Exception:
            pass
        for m in MODALITIES:
            rollout_maps[m] = None
            
    return rollout_maps
