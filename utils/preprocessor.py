"""
Preprocessing — image transforms and CSV validation.
"""

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import io
import streamlit as st
from typing import Optional, Tuple, Dict, List

# ── ImageNet normalisation (standard for corneal images) ────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(uploaded_file) -> Optional[torch.Tensor]:
    """
    Load an uploaded Streamlit file, apply standard ophthalmic preprocessing,
    and return a [1, 3, H, W] tensor.
    """
    try:
        img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0)  # [1, 3, 224, 224]
        return tensor
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None


def preprocess_image_pil(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image directly."""
    img = pil_img.convert("RGB")
    return TRANSFORM(img).unsqueeze(0)


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a preprocessed tensor back to a displayable uint8 numpy array."""
    t = tensor.squeeze(0).cpu()
    # Inverse normalise
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = t * std + mean
    t = t.clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ── CSV Validation ───────────────────────────────────────────────────────────

def validate_csv(
    uploaded_csv,
    tabular_model,
) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """
    Parse and validate an uploaded CSV file.

    Returns:
        df          — parsed single-row DataFrame (or None on critical failure)
        errors      — list of error strings
        warnings    — list of warning strings
    """
    errors: List[str] = []
    warnings: List[str] = []

    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        errors.append(f"Cannot parse CSV: {e}")
        return None, errors, warnings

    if df.shape[0] == 0:
        errors.append("CSV is empty — no patient row found.")
        return None, errors, warnings

    if df.shape[0] > 1:
        warnings.append(f"CSV has {df.shape[0]} rows. Only the first row will be used.")
        df = df.iloc[[0]]

    # Compare against model feature names if available
    model_features: Optional[List[str]] = None
    if hasattr(tabular_model, "feature_names_in_"):
        model_features = list(tabular_model.feature_names_in_)
    elif hasattr(tabular_model, "named_steps"):
        for step in tabular_model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                model_features = list(step.feature_names_in_)
                break

    if model_features is not None:
        csv_cols = set(df.columns.tolist())
        model_cols = set(model_features)

        missing = model_cols - csv_cols
        extra = csv_cols - model_cols

        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")
        if extra:
            warnings.append(f"Extra columns (will be ignored): {sorted(extra)}")

        if not missing:
            df = df[model_features]  # reorder to match model expectation

    # Check for nulls
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        warnings.append(f"Null values detected in: {null_cols}. Will be imputed with column median.")
        df = df.fillna(df.median(numeric_only=True))

    # Check all numeric
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        errors.append(f"Non-numeric columns detected: {non_numeric}. All features must be numeric.")

    if errors:
        return None, errors, warnings

    return df, errors, warnings
