import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from mm_architecture import MODALITIES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def preprocess_tensor(pil_img: Image.Image) -> torch.Tensor:
    return TRANSFORM(pil_img).unsqueeze(0)


def tensor_to_display_u8(tensor_chw: torch.Tensor) -> np.ndarray:
    """tensor [3,H,W] normalized -> uint8 HWC RGB."""
    t = tensor_chw.detach().cpu().float()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = (t * std + mean).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def numpy_bgr_from_pil(pil_img: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def validate_csv(
    uploaded_csv: io.BytesIO | bytes,
    tabular_model,
) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """
    Same rules as utils/preprocessor.validate_csv (without Streamlit).
    `uploaded_csv` can be BytesIO or raw bytes.
    """
    errors: List[str] = []
    warnings: List[str] = []

    bio = uploaded_csv if isinstance(uploaded_csv, io.BytesIO) else io.BytesIO(uploaded_csv)
    bio.seek(0)

    try:
        df = pd.read_csv(bio)
    except Exception as e:
        errors.append(f"Cannot parse CSV: {e}")
        return None, errors, warnings

    if df.shape[0] == 0:
        errors.append("CSV is empty — no patient row found.")
        return None, errors, warnings

    if df.shape[0] > 1:
        warnings.append(f"CSV has {df.shape[0]} rows. Only the first row will be used.")
        df = df.iloc[[0]]

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
            df = df[model_features]

    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        warnings.append(f"Null values detected in: {null_cols}. Will be imputed with column median.")
        df = df.fillna(df.median(numeric_only=True))

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        errors.append(f"Non-numeric columns detected: {non_numeric}. All features must be numeric.")

    if errors:
        return None, errors, warnings

    return df, errors, warnings
