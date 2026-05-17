from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def _vit_attention_maps(model: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Return [L,H,HW,HW] attention if a ViT-style block is detected; else None."""
    blocks = []
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        blocks = list(model.encoder.layers)
    elif hasattr(model, "blocks"):
        blocks = list(model.blocks)
    if not blocks:
        return None

    attn_weights: List[torch.Tensor] = []

    hooks = []

    def make_hook():
        def hook(module, inp, out):
            # timm / torch vit: out is tuple (attn_out, attn_weights) sometimes
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_weights.append(out[1].detach())
            elif hasattr(module, "attn_drop") and hasattr(module, "qkv"):
                # fallback: skip
                return

        return hook

    for b in blocks:
        if hasattr(b, "attn"):
            hooks.append(b.attn.register_forward_hook(make_hook()))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    if not attn_weights:
        return None
    # stack layers: [L,B,H,N,N] -> take mean batch/heads
    a = torch.stack(attn_weights, dim=0).float()
    if a.dim() == 5:
        a = a.mean(dim=(1, 2))
    return a


def attention_rollout(attn: torch.Tensor, discard_ratio: float = 0.0) -> np.ndarray:
    """
    attn: [L, N, N] attention matrices per layer (already averaged over heads).
    """
    result = torch.eye(attn.shape[-1], device=attn.device, dtype=attn.dtype)
    for a in attn:
        a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype) * 0.001
        a = a / a.sum(dim=-1, keepdim=True)
        if discard_ratio > 0:
            flat = a.view(-1)
            _, idx = torch.topk(flat, int(flat.numel() * discard_ratio), largest=False)
            flat[idx] = 0
            a = flat.view_as(a)
        result = torch.matmul(a, result)
    vis = result[0, 1:]  # CLS -> patches (skip CLS row 0 for map)
    vis = vis - vis.min()
    vis = vis / (vis.max() + 1e-8)
    n = vis.numel()
    side = int(np.sqrt(n))
    if side * side != n:
        return vis.detach().cpu().numpy().reshape(-1)
    return vis.detach().cpu().numpy().reshape(side, side)


def run_attention_rollout(model: nn.Module, input_tensor: torch.Tensor) -> Optional[Dict[str, str]]:
    import base64
    import cv2

    device = next(model.parameters()).device
    x = input_tensor.detach().to(device)
    attn = _vit_attention_maps(model, x)
    if attn is None:
        return None
    if attn.dim() == 4:
        attn = attn.mean(dim=1)
    map2d = attention_rollout(attn)
    if map2d.ndim != 2:
        return None
    map_u8 = (np.clip(map2d, 0, 1) * 255).astype(np.uint8)
    map_u8 = cv2.resize(map_u8, (224, 224), interpolation=cv2.INTER_CUBIC)
    _, buf = cv2.imencode(".png", map_u8)
    return {"heatmap_b64": base64.b64encode(buf.tobytes()).decode("ascii")}
