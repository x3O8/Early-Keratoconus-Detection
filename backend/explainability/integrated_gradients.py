from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def run_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    steps: int = 32,
) -> Optional[Dict[str, str]]:
    """Integrated gradients attribution map, returned as base64 heatmap."""
    import base64
    import cv2

    try:
        from captum.attr import IntegratedGradients
    except Exception:
        return None

    device = next(model.parameters()).device
    x = input_tensor.detach().to(device).requires_grad_(True)
    model.eval()

    def forward_fn(inp):
        out = model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("logits", next(iter(out.values())))
        return out

    ig = IntegratedGradients(forward_fn)
    attr, _ = ig.attribute(
        x,
        target=int(target_class),
        n_steps=int(steps),
        internal_batch_size=1,
        return_convergence_delta=True,
    )
    attr = attr.detach().squeeze(0).cpu().numpy()
    attr = np.mean(np.abs(attr), axis=0)
    a_min, a_max = float(attr.min()), float(attr.max())
    if a_max - a_min < 1e-8:
        sal = np.zeros_like(attr, dtype=np.float32)
    else:
        sal = (attr - a_min) / (a_max - a_min + 1e-8)

    sal_u8 = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", sal_u8)
    return {"heatmap_b64": base64.b64encode(buf.tobytes()).decode("ascii")}
