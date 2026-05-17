"""
Model Loader — automatically discovers and loads .pth and .pkl models
from the working directory and common subdirectories.
"""

import os
import glob
import pickle
import joblib
import torch
import torch.nn as nn
import torchvision.models as tv_models
import streamlit as st
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

MODALITIES = ["CT_A", "EC_A", "EC_P", "Elv_A", "Elv_P", "Sag_A", "Sag_P"]
NUM_MODALITIES = len(MODALITIES)


def get_device():
    """Return best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0], x.shape[1], 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


class MultiHeadModalityAttention(nn.Module):
    def __init__(self, feature_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert feature_dim % n_heads == 0
        self.n_heads    = n_heads
        self.head_dim   = feature_dim // n_heads
        self.scale      = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj   = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj   = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(feature_dim)
        self.modality_gate = nn.Parameter(torch.ones(1, 7, 1))

    def forward(self, x):
        B, M, D = x.shape
        x_norm = self.norm(x)

        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)

        def split_heads(t):
            t = t.view(B, M, self.n_heads, self.head_dim)
            return t.permute(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn, dim=-1)
        self.last_attn_probs = attn_probs.detach()
        attn_drop = self.attn_drop(attn_probs)

        out = torch.matmul(attn_drop, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, M, D)
        out = self.out_proj(out)

        out = x + out
        gate = torch.sigmoid(self.modality_gate)
        out  = out * gate

        attn_weights = attn_probs.mean(dim=1).mean(dim=1)
        return out.mean(dim=1), attn_weights


class MultiModalKeratoconusModel(nn.Module):
    def __init__(self, num_classes: int = 2, num_modalities: int = 7,
                 proj_dim: int = 320, dropout: float = 0.3,
                 drop_path_rate: float = 0.1, n_attn_heads: int = 4):
        super().__init__()
        self.num_modalities = num_modalities
        self.proj_dim = proj_dim

        base = efficientnet_b0(weights=None)
        self.feature_dim   = base.classifier[1].in_features  # 1280
        self.backbone      = base.features
        self.backbone_pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.drop_path = DropPath(drop_prob=drop_path_rate)

        self.attention_fusion = MultiHeadModalityAttention(
            feature_dim=proj_dim,
            n_heads=n_attn_heads,
            dropout=0.1
        )

        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.proj, self.head, self.attention_fusion]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, images: list):
        # images: list of tensors [B, 3, 224, 224]
        x = torch.stack(images, dim=1) # [B, 7, 3, 224, 224]
        B, M, C, H, W = x.shape
        
        spatial_feats = []
        for i in range(M):
            feat = self.backbone(x[:, i])
            spatial_feats.append(feat)
            
        feats = torch.stack(spatial_feats, dim=1) # [B, 7, 1280, 7, 7]
        feats_flat = feats.view(B * M, self.feature_dim, feats.shape[-2], feats.shape[-1])
        pooled = self.backbone_pool(feats_flat).flatten(1)
        
        proj = self.proj(pooled)
        proj = proj.view(B, M, self.proj_dim)
        
        dropped = self.drop_path(proj)
        fused, attn_weights = self.attention_fusion(dropped)
        logits = self.head(fused)
        
        return logits, attn_weights, spatial_feats


def _find_files(extensions, search_dirs=None):
    """Search working directory tree for files with given extensions."""
    if search_dirs is None:
        search_dirs = ["."]
    found = []
    for d in search_dirs:
        for ext in extensions:
            found.extend(glob.glob(os.path.join(d, "**", f"*{ext}"), recursive=True))
    return list(set(found))


@st.cache_resource(show_spinner=False)
def load_image_model():
    """
    Auto-discover and load the multi-modal image model.
    Falls back to an untrained architecture if no .pth file found.
    """
    device = get_device()
    model = MultiModalKeratoconusModel(
        num_modalities=NUM_MODALITIES,
        proj_dim=320,
        n_attn_heads=4,
        num_classes=2,
    ).to(device)
    model.eval()

    # Search for checkpoint
    pth_files = _find_files([".pth", ".pt"], search_dirs=[".", "models", "checkpoints", "weights"])

    loaded_checkpoint = False
    loaded_path = None
    for pth_path in sorted(pth_files):
        try:
            state = torch.load(pth_path, map_location=device, weights_only=False)
            # Handle various checkpoint formats
            if isinstance(state, dict):
                sd = state.get("model_state_dict") or state.get("state_dict") or state
            else:
                sd = state
            # Try strict=False to handle partial weights gracefully
            model.load_state_dict(sd, strict=False)
            loaded_checkpoint = True
            loaded_path = Path(pth_path).name
            break
        except Exception:
            continue  # try next file

    return model, device, loaded_checkpoint, loaded_path


@st.cache_resource(show_spinner=False)
def load_tabular_model():
    """
    Auto-discover and load the tabular ML model from .pkl/.joblib files.
    Falls back to a lightweight sklearn pipeline in demo mode.
    """
    pkl_files = _find_files([".pkl", ".joblib"], search_dirs=[".", "models", "pipelines"])

    for pkl_path in sorted(pkl_files):
        try:
            obj = joblib.load(pkl_path)
            # Must have predict_proba
            if hasattr(obj, "predict_proba"):
                return obj, True, Path(pkl_path).name
        except Exception:
            try:
                with open(pkl_path, "rb") as f:
                    obj = pickle.load(f)
                if hasattr(obj, "predict_proba"):
                    return obj, True, Path(pkl_path).name
            except Exception:
                continue

    # Demo fallback
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    demo_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42)),
    ])
    rng = np.random.default_rng(42)
    X_dummy = rng.standard_normal((100, 10))
    y_dummy = (rng.random(100) > 0.5).astype(int)
    demo_model.fit(X_dummy, y_dummy)
    demo_model.feature_names_in_ = [f"feature_{i}" for i in range(10)]
    return demo_model, False, None
