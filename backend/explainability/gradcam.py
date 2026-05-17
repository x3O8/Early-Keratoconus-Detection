from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_last_conv2d(model: nn.Module) -> Optional[nn.Conv2d]:
    last: Optional[nn.Conv2d] = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def find_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """
    Prefer the last conv in the main feature trunk (ResNet layer4, VGG features, EfficientNet features).
    """
    if hasattr(model, "layer4") and len(list(model.layer4.modules())) > 0:
        convs = [m for m in model.layer4.modules() if isinstance(m, nn.Conv2d)]
        if convs:
            return convs[-1]
    if hasattr(model, "features"):
        convs = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
        if convs:
            return convs[-1]
    c = find_last_conv2d(model)
    if c is None:
        raise RuntimeError("No Conv2d found for GradCAM")
    return c


class GradCAM:
    """
    Standard Grad-CAM with correct activation/gradient shapes and positive normalization.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _save_activation(self, module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        g = grad_out[0]
        if g is None:
            return
        self.gradients = g.detach()

    def compute(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        out = self.model(input_tensor)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        elif isinstance(out, dict):
            logits = out.get("logits", next(iter(out.values())))
        else:
            logits = out

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations")

        acts = self.activations
        grads = self.gradients

        if acts.dim() != 4 or grads.dim() != 4:
            raise RuntimeError(f"Expected NCHW activations; got acts={acts.shape}, grads={grads.shape}")

        if acts.shape != grads.shape:
            # align spatial/channel if needed (rare depthwise cases)
            grads = F.adaptive_avg_pool2d(grads, output_size=acts.shape[-2:])

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=False)
        cam = F.relu(cam)

        cam_np = cam.squeeze(0).detach().float().cpu().numpy()
        cmin, cmax = float(cam_np.min()), float(cam_np.max())
        if cmax - cmin < 1e-8:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)
        else:
            cam_np = (cam_np - cmin) / (cmax - cmin + 1e-8)
        return cam_np.astype(np.float32)


def resize_cam_to_image(cam: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if cam.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    cam_u8 = np.clip(cam * 255.0, 0, 255).astype(np.uint8)
    resized = cv2.resize(cam_u8, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    rmin, rmax = float(resized.min()), float(resized.max())
    if rmax - rmin < 1e-8:
        return np.zeros_like(resized, dtype=np.float32)
    return (resized - rmin) / (rmax - rmin + 1e-8)


def overlay_heatmap_bgr(
    image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45, colormap: int = None
) -> np.ndarray:
    import cv2

    if colormap is None:
        colormap = cv2.COLORMAP_TURBO
    h, w = image_bgr.shape[:2]
    cam_r = resize_cam_to_image(cam, h, w)
    heat = cv2.applyColorMap((np.clip(cam_r, 0, 1) * 255).astype(np.uint8), colormap)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heat, alpha, 0)
    return overlay


def run_gradcam(
    model: nn.Module, input_tensor: torch.Tensor, target_class: int, image_bgr: np.ndarray, alpha: float
) -> Dict[str, str]:
    """Returns dict with base64 PNG strings."""
    import base64
    import cv2

    device = next(model.parameters()).device
    x = input_tensor.to(device)
    x = x.detach().requires_grad_(True)

    layer = find_gradcam_target_layer(model)
    gc = GradCAM(model, layer)
    try:
        cam = gc.compute(x, int(target_class))
    finally:
        gc.remove()

    overlay_bgr = overlay_heatmap_bgr(image_bgr, cam, alpha=alpha)
    _, buf_o = cv2.imencode(".png", overlay_bgr)
    _, buf_h = cv2.imencode(".png", (np.clip(resize_cam_to_image(cam, image_bgr.shape[0], image_bgr.shape[1]), 0, 1) * 255).astype(np.uint8))

    return {
        "overlay_b64": base64.b64encode(buf_o.tobytes()).decode("ascii"),
        "heatmap_b64": base64.b64encode(buf_h.tobytes()).decode("ascii"),
        "target_layer": layer.__class__.__name__,
    }
