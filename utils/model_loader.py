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


# ────────────────────────────────────────────────────────────────────────────
# Image Model Architecture
# ────────────────────────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """Single-modality ResNet encoder with optional attention."""
    def __init__(self, embed_dim=256, pretrained=False):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-2])  # drop avgpool+fc
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        feat = self.features(x)          # [B, 512, H, W]
        pooled = self.pool(feat).flatten(1)
        return self.proj(pooled), feat   # embed + spatial features for Grad-CAM


class MultiModalKeratoconusModel(nn.Module):
    """
    Processes 7 ophthalmic modalities independently, then fuses via
    cross-modal attention and classifies.
    """
    def __init__(self, num_modalities=7, embed_dim=256, num_heads=4, num_classes=2):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim

        # Per-modality encoders
        self.encoders = nn.ModuleList([
            ModalityEncoder(embed_dim) for _ in range(num_modalities)
        ])

        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Modality-level attention weights (for importance visualization)
        self.modality_attn = nn.Linear(embed_dim, 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: list):
        """
        images: list of tensors [B, 3, 224, 224], one per modality.
        Returns: logits, modality_weights, spatial_features (for Grad-CAM).
        """
        embeds = []
        spatial_feats = []
        for i, img in enumerate(images):
            embed, spatial = self.encoders[i](img)
            embeds.append(embed.unsqueeze(1))   # [B, 1, D]
            spatial_feats.append(spatial)

        tokens = torch.cat(embeds, dim=1)       # [B, M, D]
        tokens = self.transformer(tokens)        # [B, M, D]

        # Soft attention over modalities
        attn_scores = self.modality_attn(tokens).squeeze(-1)   # [B, M]
        attn_weights = torch.softmax(attn_scores, dim=-1)      # [B, M]

        # Weighted sum
        context = (tokens * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        logits = self.classifier(context)

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
        embed_dim=256,
        num_heads=4,
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
