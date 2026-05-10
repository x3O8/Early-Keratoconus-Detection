# KeratoScan AI — Keratoconus Detection Dashboard

A production-grade Streamlit dashboard for AI-assisted keratoconus detection using
multi-modal ophthalmic image analysis fused with clinical tabular data.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Place your model files in the project root or subdirectories:
#    - Multi-modal image model: any .pth or .pt file
#    - Tabular model:           any .pkl or .joblib file
#    Both are auto-discovered — no filenames to configure.

# 3. Launch
streamlit run app.py
```

---

## System Architecture

```
7 Corneal Images (CT_A, EC_A, EC_P, Elv_A, Elv_P, Sag_A, Sag_P)
         │
         ▼
Multi-Modal Transformer (ResNet-18 encoders + cross-attention)
         │
         ├── Attention Maps (modality importance)
         └── Grad-CAM Heatmaps (spatial explanations)

1-Row Clinical CSV
         │
         ▼
Tabular ML Pipeline (sklearn compatible: XGBoost / RF / LogReg)
         │
         └── Feature Importance

         ┌────────────────────┐
         │  Weighted Ensemble │
         │  (configurable)    │
         └────────────────────┘
                  │
                  ▼
         Final Keratoconus Prediction
                  │
                  ▼
         Gemini 1.5 Flash LLM
                  │
                  ▼
         Clinical Report + Treatment Suggestions
```

---

## Image Modalities

| Key   | Full Name                        |
|-------|----------------------------------|
| CT_A  | Corneal Thickness — Anterior     |
| EC_A  | Epithelial Curvature — Anterior  |
| EC_P  | Epithelial Curvature — Posterior |
| Elv_A | Elevation Map — Anterior         |
| Elv_P | Elevation Map — Posterior        |
| Sag_A | Sagittal Curvature — Anterior    |
| Sag_P | Sagittal Curvature — Posterior   |

Missing modalities are substituted with zero tensors — partial inputs are supported.

---

## Model Auto-Discovery

The app scans the working directory (and subdirectories) for:

- **`.pth` / `.pt`** → loaded as image model checkpoint  
  Supports: `model_state_dict`, `state_dict`, or raw `OrderedDict` formats.

- **`.pkl` / `.joblib`** → loaded as tabular model  
  Must implement `predict_proba`. Compatible with sklearn, XGBoost, LightGBM, etc.

If no files are found, both models run in **demo mode** with random initialisation
(for UI development and testing).

---

## CSV Format

```
feature1,feature2,feature3,...
1.23,4.56,7.89,...
```

- First row: column headers
- Second row: patient data values
- All values must be numeric
- Column names must match the tabular model's `feature_names_in_`

---

## Preprocessing

Images are preprocessed identically to ImageNet-pretrained models:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## Ensemble Fusion

```
final_prob = image_weight × p_image + tabular_weight × p_tabular
```

Weights are normalised to sum to 1.0. Default: 65% image / 35% tabular.
Adjust in the sidebar slider.

---

## Gemini Integration

Enter your Gemini API key (from [aistudio.google.com](https://aistudio.google.com)) in the sidebar.

- Key is stored in Streamlit session state only
- Never logged or persisted
- Uses `gemini-1.5-flash` model for speed and cost efficiency
- Prompt includes model probabilities, severity, top modalities, and top features

---

## Explainability

| Feature               | Source                    |
|-----------------------|---------------------------|
| Grad-CAM heatmaps     | Per-modality encoder (last conv layer) |
| Modality importance   | Cross-modal transformer attention weights |
| Feature importance    | Tabular model `feature_importances_` or `coef_` |

---

## Disclaimer

> **This tool is for research and educational use only.**  
> It is not a CE-marked or FDA-cleared medical device.  
> All outputs must be reviewed by a qualified ophthalmologist.  
> Never use this system as the sole basis for clinical decisions.
