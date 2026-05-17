"""
Load 7-modality multimodal PyTorch model + tabular sklearn model (matches Streamlit discovery).
"""

from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import joblib
import torch

from config import MODELS_DIR
from mm_architecture import MultiModalKeratoconusModel, NUM_MODALITIES


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_pth_files() -> List[str]:
    patterns = [
        os.path.join(MODELS_DIR, "*.pth"),
        os.path.join(MODELS_DIR, "*.pt"),
    ]
    found: List[str] = []
    for p in patterns:
        found.extend(glob.glob(p))
    return sorted(set(found))


def _find_tabular_files() -> List[str]:
    found: List[str] = []
    for ext in ("*.pkl", "*.joblib"):
        found.extend(glob.glob(os.path.join(MODELS_DIR, ext)))
    priority_names = ("tabular", "model2", "clinical", "tabular_model")
    def sort_key(p: str) -> Tuple[int, str]:
        base = os.path.basename(p).lower()
        for i, name in enumerate(priority_names):
            if base.startswith(name):
                return (i, p)
        return (len(priority_names), p)
    return sorted(set(found), key=sort_key)


def _multimodal_state_dict_score(sd: dict, model: MultiModalKeratoconusModel) -> float:
    msd = model.state_dict()
    matched = sum(1 for k in msd if k in sd and msd[k].shape == sd[k].shape)
    return matched / max(1, len(msd))


def load_multimodal_from_disk(device: torch.device) -> Tuple[MultiModalKeratoconusModel, bool, Optional[str], dict]:
    meta: dict = {"architecture": "MultiModalKeratoconusModel", "num_modalities": NUM_MODALITIES}
    candidates = _find_pth_files()
    best_score = -1.0
    best_sd: Optional[dict] = None
    best_path: Optional[str] = None

    for path in candidates:
        try:
            raw = torch.load(path, map_location=device, weights_only=False)
            sd = raw
            if isinstance(raw, dict):
                sd = raw.get("model_state_dict") or raw.get("state_dict") or raw
            if not isinstance(sd, dict):
                continue
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            if not any(k.startswith("encoders.") for k in sd):
                continue
            probe = MultiModalKeratoconusModel(
                num_modalities=NUM_MODALITIES,
                embed_dim=256,
                num_heads=4,
                num_classes=2,
            ).to(device)
            probe.load_state_dict(sd, strict=False)
            score = _multimodal_state_dict_score(sd, probe)
            if score > best_score:
                best_score = score
                best_sd = sd
                best_path = path
        except Exception:
            continue

    model = MultiModalKeratoconusModel(
        num_modalities=NUM_MODALITIES,
        embed_dim=256,
        num_heads=4,
        num_classes=2,
    ).to(device)

    if best_sd is not None and best_score > 0.05:
        model.load_state_dict(best_sd, strict=False)
        model.eval()
        return (
            model,
            True,
            best_path,
            {**meta, "checkpoint": os.path.basename(best_path) if best_path else None, "match_score": best_score},
        )

    model.eval()
    return model, False, None, {**meta, "note": "No compatible multimodal .pth in models/ — using random init"}


def _demo_tabular():
    from sklearn.linear_model import LogisticRegression

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    X_dummy = rng.standard_normal((100, 10))
    y_dummy = (rng.random(100) > 0.5).astype(int)
    cols = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X_dummy, columns=cols)
    clf = LogisticRegression(random_state=42, max_iter=500)
    clf.fit(df, y_dummy)
    return clf


def load_tabular_from_disk() -> Tuple[Any, bool, Optional[str], dict]:
    for path in _find_tabular_files():
        try:
            try:
                obj = joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
            if hasattr(obj, "predict_proba") or hasattr(obj, "decision_function"):
                return obj, True, path, {"kind": "sklearn", "path": os.path.basename(path)}
        except Exception:
            continue
    m = _demo_tabular()
    return m, False, None, {"kind": "sklearn", "note": "Demo LogisticRegression(10 features)"}


@dataclass
class AppModelCache:
    device: torch.device = field(default_factory=get_device)
    image_model: Optional[MultiModalKeratoconusModel] = None
    tabular_model: Any = None
    image_path: Optional[str] = None
    tabular_path: Optional[str] = None
    image_meta: dict = field(default_factory=dict)
    tabular_meta: dict = field(default_factory=dict)
    _errors: List[str] = field(default_factory=list)

    def refresh(self) -> None:
        self._errors = []
        self.device = get_device()
        try:
            self.image_model, loaded, path, meta = load_multimodal_from_disk(self.device)
            self.image_path = path
            self.image_meta = meta
            if not loaded:
                self._errors.append("Image model: using uninitialized multimodal weights (add a matching .pth to backend/models/).")
        except Exception as e:
            self._errors.append(f"Image model load: {e}")
            self.image_model = MultiModalKeratoconusModel().to(self.device).eval()
            self.image_path = None
            self.image_meta = {"error": str(e)}

        try:
            self.tabular_model, tloaded, tpath, tmeta = load_tabular_from_disk()
            self.tabular_path = tpath
            self.tabular_meta = tmeta
            if not tloaded:
                self._errors.append("Tabular model: using built-in demo sklearn pipeline (upload CSV with feature_0…feature_9 or train columns).")
        except Exception as e:
            self._errors.append(f"Tabular model load: {e}")
            self.tabular_model = _demo_tabular()
            self.tabular_path = None
            self.tabular_meta = {"error": str(e)}


cache = AppModelCache()