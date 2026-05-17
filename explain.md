# KeratoScan AI — Complete System Documentation

> **Version:** v1.0 · Research Build  
> **Stack:** Streamlit · PyTorch · scikit-learn · Gemini 2.0 Flash  
> **Purpose:** Multi-modal AI-assisted keratoconus detection (research/educational use only)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Input Data](#3-input-data)
4. [Model 1 — Multi-Modal Image Model](#4-model-1--multi-modal-image-model)
5. [Model 2 — Tabular Clinical Model](#5-model-2--tabular-clinical-model)
6. [Ensemble Fusion](#6-ensemble-fusion)
7. [Explainability — Grad-CAM](#7-explainability--grad-cam)
8. [Explainability — Feature Importance](#8-explainability--feature-importance)
9. [RAG System](#9-rag-system)
10. [LLM Report Generation](#10-llm-report-generation)
11. [UI Tabs & Workflow](#11-ui-tabs--workflow)
12. [Output Results Reference](#12-output-results-reference)
13. [File Structure](#13-file-structure)
14. [Dependencies](#14-dependencies)

---

## 1. System Overview

KeratoScan AI is a multi-modal, AI-powered clinical decision support system for detecting **keratoconus (KC)** — a progressive corneal ectatic disorder. It fuses two independent AI streams:

| Stream | Input | Model Type |
|--------|-------|-----------|
| Image Stream | Up to 7 corneal imaging modalities (PNG/JPG) | Custom EfficientNet-B0 + Multi-Head Attention |
| Tabular Stream | Patient topography/biometry CSV | Scikit-learn compatible (XGBoost / pipeline) |

Both streams produce independent probability estimates which are fused via **weighted ensemble**. The result is then explained through **Grad-CAM heatmaps**, **feature importance charts**, and a **RAG-augmented Gemini LLM report**.

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│  [CT_A] [EC_A] [EC_P] [Elv_A] [Elv_P] [Sag_A] [Sag_P]  + CSV      │
└────────────────────────┬─────────────────────────┬───────────────────┘
                         │                         │
              ┌──────────▼──────────┐   ┌──────────▼──────────┐
              │  IMAGE MODEL        │   │  TABULAR MODEL       │
              │  EfficientNet-B0    │   │  XGBoost / sklearn   │
              │  Shared backbone    │   │  pipeline            │
              │  ↓                  │   │  ↓                   │
              │  Multi-Head Attn    │   │  predict_proba()     │
              │  Fusion (7→1)       │   │  feature_importances_│
              │  ↓                  │   │  ↓                   │
              │  P(KC | image)      │   │  P(KC | tabular)     │
              └──────────┬──────────┘   └──────────┬──────────┘
                         │                         │
              ┌──────────▼─────────────────────────▼──────────┐
              │           WEIGHTED ENSEMBLE FUSION             │
              │   P(KC) = w_img × P_img + w_tab × P_tab        │
              │   Default: w_img=0.65, w_tab=0.35              │
              └──────────────────────┬─────────────────────────┘
                                     │
              ┌──────────────────────▼─────────────────────────┐
              │              EXPLAINABILITY LAYER               │
              │  ├── Grad-CAM heatmaps (per modality)          │
              │  ├── Modality attention weights                 │
              │  └── Tabular feature importance                 │
              └──────────────────────┬─────────────────────────┘
                                     │
              ┌──────────────────────▼─────────────────────────┐
              │           RAG + LLM REPORT LAYER                │
              │  ├── TF-IDF retrieval from RAG Files/           │
              │  ├── Top-5 relevant clinical passages           │
              │  └── Gemini 2.0 Flash → structured report       │
              └────────────────────────────────────────────────┘
```

---

## 3. Input Data

### 3.1 Corneal Image Modalities

The system accepts up to **7 corneal imaging maps**, each uploaded as a separate PNG/JPG:

| Code | Full Name | What It Shows | Key KC Indicator |
|------|-----------|---------------|-----------------|
| `CT_A` | Corneal Thickness (Anterior) | Full-thickness pachymetry map | Thinnest point < 480 µm, inferior displacement |
| `EC_A` | Epithelial Curvature (Anterior) | Epithelial thickness distribution | "Donut" thinning pattern at cone apex |
| `EC_P` | Epithelial Curvature (Posterior) | Bowman's layer interface curvature | Focal steepening, irregular pattern |
| `Elv_A` | Elevation Map (Anterior) | Anterior surface height above BFS | Focal island > +15 µm |
| `Elv_P` | Elevation Map (Posterior) | Posterior surface height above BFS | **Most sensitive** — elevation > +15 µm |
| `Sag_A` | Sagittal Curvature (Anterior) | Axial curvature in dioptres | Kmax > 47.2 D, inferior steepening |
| `Sag_P` | Sagittal Curvature (Posterior) | Posterior axial curvature | Focal posterior steepening |

**Missing modalities:** If a modality is not uploaded, a **zero-filled tensor** of shape `[1, 3, 224, 224]` is substituted. The model is robust to partial inputs.

### 3.2 Image Preprocessing

Each uploaded image goes through:

```python
transforms.Compose([
    transforms.Resize((224, 224)),       # Resize to EfficientNet input size
    transforms.ToTensor(),               # PIL → float32 tensor [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      # ImageNet mean
        std=[0.229, 0.224, 0.225]        # ImageNet std
    )
])
```

Output tensor shape: `[1, 3, 224, 224]`

### 3.3 Clinical CSV (Tabular Input)

A single-row CSV with numeric topography/biometry features. Requirements:
- **Header row** + **1 data row** (multi-row CSVs: only first row used)
- **All numeric** columns (non-numeric columns cause a validation error)
- **Column names** must match the trained tabular model's `feature_names_in_`
- **Null values** → imputed with column median and warned

---

## 4. Model 1 — Multi-Modal Image Model

### 4.1 Architecture: `MultiModalKeratoconusModel`

**File:** `utils/model_loader.py`

#### Stage 1 — Shared Backbone (EfficientNet-B0)

All 7 modality images share **the same EfficientNet-B0 backbone** (weights loaded from `model1.pth`):

```
Input: [B, 3, 224, 224]  (one modality)
  → EfficientNet-B0 features  (backbone.features)
  → AdaptiveAvgPool2d(1)
  → Flatten
Output: [B, 1280]  (feature vector per modality)
```

Processing all 7 modalities in a loop produces a stacked tensor: `[B, 7, 1280]`

#### Stage 2 — Projection Head

```
[B, 7, 1280]
  → LayerNorm(1280)
  → Linear(1280 → 320)
  → GELU activation
  → Dropout(0.15)
Output: [B, 7, 320]
```

#### Stage 3 — DropPath Regularisation

Stochastic depth regularisation applied during training (`drop_prob=0.1`). Disabled at inference.

#### Stage 4 — Multi-Head Modality Attention (`MultiHeadModalityAttention`)

The core fusion module. Uses **4-head self-attention** across the 7 modality tokens:

```
Input: [B, 7, 320]  (7 modality feature vectors)

Q = Linear(320 → 320)  split into 4 heads of dim 80
K = Linear(320 → 320)
V = Linear(320 → 320)

Attention weights = softmax(QK^T / sqrt(80))   → [B, 4, 7, 7]
Context = Attention_weights × V                 → [B, 7, 320]

Gated output = (residual + context) × sigmoid(modality_gate)

Mean-pool across 7 modalities → [B, 320]
Modality weights = mean of attention probs across heads → [B, 7]
```

The **modality weights** `[B, 7]` are the cross-modal attention values — they quantify which imaging modality drove the prediction for this specific case. These are used in the Explainability tab.

#### Stage 5 — Classification Head

```
[B, 320]
  → LayerNorm(320)
  → Linear(320 → 160)
  → GELU
  → Dropout(0.3)
  → Linear(160 → 2)
Output: logits [B, 2]  → softmax → [P(Normal), P(KC)]
```

> **Label convention:** The model was trained with Keratoconus=0, Normal=1 internally. The UI remaps to Normal=0, Keratoconus=1 during inference.

### 4.2 Model Discovery & Loading

`load_image_model()` (cached with `@st.cache_resource`) automatically:
1. Scans `./`, `./models/`, `./checkpoints/`, `./weights/` for `*.pth` and `*.pt` files
2. Tries loading each with `torch.load(..., weights_only=False)`
3. Handles multiple checkpoint formats: `model_state_dict`, `state_dict`, or raw state dict
4. Uses `strict=False` for partial weight loading
5. Falls back to **untrained architecture** (demo mode) if no checkpoint found

### 4.3 Image Inference Output

```python
{
  "probs": np.array([P_normal, P_kc]),     # 2-class probabilities
  "predicted_class": int,                  # 0=Normal, 1=KC
  "predicted_label": str,                  # "Normal" or "Keratoconus"
  "confidence": float,                     # max(probs)
  "keratoconus_prob": float,               # P_kc
  "normal_prob": float,                    # P_normal
  "modality_weights": {                    # attention weights per modality
      "CT_A": 0.18, "EC_A": 0.12, ...
  },
  "spatial_feats": [...],                  # raw feature maps (for Grad-CAM)
  "inputs": [...]                          # input tensors (for Grad-CAM)
}
```

---

## 5. Model 2 — Tabular Clinical Model

### 5.1 Model Type

Any **scikit-learn compatible** estimator with `predict_proba()`. The bundled model (`model2.pkl`) is typically an XGBoost or gradient-boosting pipeline with preprocessing steps.

**File:** `utils/inference.py` → `run_tabular_inference()`

### 5.2 Model Discovery & Loading

`load_tabular_model()` (cached) scans for `*.pkl` and `*.joblib` files and loads the first one with `predict_proba`. If none found, falls back to a **demo logistic regression** on random data.

### 5.3 Tabular Inference Output

```python
{
  "probs": np.array([P_normal, P_kc]),
  "predicted_class": int,
  "predicted_label": str,
  "confidence": float,
  "keratoconus_prob": float,
  "normal_prob": float,
  "feature_importance": {
      "Kmax": 0.23, "MinPachy": 0.18, ...  # normalised to sum=1
  }
}
```

### 5.4 Feature Importance Extraction

The system tries multiple strategies in order:
1. `clf.feature_importances_` — tree-based models (XGBoost, RF, GBM)
2. `abs(clf.coef_[0])` — linear models (logistic regression, SVM)
3. `np.random.dirichlet(ones)` — fallback for unsupported model types

All importances are L1-normalised to sum to 1.0.

---

## 6. Ensemble Fusion

**File:** `utils/inference.py` → `ensemble_fusion()`

### 6.1 Weighted Probability Fusion

```
w_img  = image_weight  / (image_weight + tabular_weight)
w_tab  = tabular_weight / (image_weight + tabular_weight)

P_ensemble(KC) = w_img × P_image(KC) + w_tab × P_tabular(KC)
```

Default weights: **Image 65% · Tabular 35%** (adjustable via sidebar slider 0–100%).

### 6.2 Severity Estimation

| KC Probability | Severity Label |
|---------------|---------------|
| < 40% | Unlikely |
| 40–55% | Borderline |
| 55–70% | Mild |
| 70–85% | Moderate |
| > 85% | Severe |

### 6.3 Ensemble Output

```python
{
  "predicted_label": str,           # "Normal" or "Keratoconus"
  "keratoconus_prob": float,        # fused P(KC)
  "normal_prob": float,
  "confidence": float,              # max(fused_probs)
  "severity": str,                  # "Unlikely" / "Borderline" / "Mild" / "Moderate" / "Severe"
  "image_contribution": float,      # w_img × P_image(KC)
  "tabular_contribution": float,    # w_tab × P_tabular(KC)
  "image_weight": float,
  "tabular_weight": float
}
```

---

## 7. Explainability — Grad-CAM

**File:** `utils/inference.py` → `GradCAMHook`, `compute_gradcam()`, `overlay_gradcam()`

### 7.1 What Grad-CAM Shows

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights **which spatial regions** in each corneal image most influenced the prediction. Warm (red/orange) regions = high influence; cool (blue) = low influence.

### 7.2 Implementation

Uses **PyTorch forward/backward hooks** on `model.backbone[-1]` (last convolutional block of EfficientNet-B0):

```
1. Register forward hook → saves feature maps [B, C, H, W]
2. Register backward hook → saves gradients [B, C, H, W]
3. Forward pass all 7 modalities
4. Backward pass: score.backward() for target class
5. Per modality i:
     weights = mean(gradients_i, dim=[H,W])    → [C]
     CAM = ReLU(sum(weights × activations_i))  → [H,W]
     Normalise to [0, 1]
6. Resize CAM to 224×224 and overlay on original image
```

**Overlay:** `alpha=0.45` blend of JET colormap heatmap + original image.

### 7.3 Target Class

Grad-CAM is computed for the **ensemble-predicted class** (i.e., if ensemble predicts KC, CAM highlights KC-relevant features).

---

## 8. Explainability — Feature Importance

**File:** `utils/inference.py` → `_get_feature_importance()`  
**UI:** `components/explainability_tab.py`

Bar chart (horizontal, Plotly) showing top-20 clinical features by importance, colour-graded cyan→teal by rank.

**Top-5 table** shown below the chart with inline progress bars.

---

## 9. RAG System

**File:** `utils/rag_engine.py`

### 9.1 Purpose

Retrieval-Augmented Generation (RAG) retrieves relevant passages from the clinical knowledge base and injects them into the Gemini prompt, grounding the LLM output in factual medical literature rather than relying on model weights alone.

### 9.2 Knowledge Base

Located in `RAG Files/`. Supported formats: `.txt`, `.md`, `.pdf` (pypdf).

| File | Content |
|------|---------|
| `keratoconus_clinical_guide.txt` | Definition, epidemiology, pathophysiology, classification (Amsler-Krumeich, ABCD, BAD-D), diagnosis, management, monitoring |
| `keratoconus_imaging_modalities.txt` | Technical reference for all 7 modalities: normal values, KC findings, clinical significance |
| `keratoconus_management_protocols.txt` | Severity-based algorithms, CXL protocols, contact lens selection, clinical thresholds table |

**Adding new knowledge:** Drop any `.txt`, `.md`, or `.pdf` into `RAG Files/`. The index auto-rebuilds when the file set changes.

### 9.3 Pipeline

#### Step 1 — Document Loading
```python
load_rag_documents(rag_dir) → List[(filename, full_text)]
```
Reads all supported files; PDFs extracted via `pypdf`.

#### Step 2 — Chunking
```python
chunk_text(text, source, chunk_size=400, overlap=80)
```
Splits each document into overlapping 400-word windows (80-word overlap) → `DocumentChunk` objects with `source`, `chunk_id`, and `uid`.

**Current index:** 3 documents → 10 chunks total.

#### Step 3 — TF-IDF Indexing
```python
TFIDFIndex(chunks).build()
```
Uses `sklearn.TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` + L2 normalisation. Falls back to a pure-Python BM25-style keyword index if scikit-learn is unavailable.

**Cache:** Index is stored in `st.session_state` with a content fingerprint hash; only rebuilt when documents change.

#### Step 4 — Query Generation
```python
build_query_from_results(ensemble, image_result, tabular_result, ...)
```
Constructs a rich semantic query from model outputs:
- Diagnosis label + severity string
- Severity-specific clinical terms (e.g., "cross-linking CXL surgical" for high probability)
- Top-3 modality names expanded to clinical terms (e.g., `Elv_P` → "posterior elevation ectasia protrusion")
- Top-3 feature names from tabular importance

#### Step 5 — Retrieval
```python
index.query(query_text, top_k=5, min_score=0.05) → [(chunk, score)]
```
Cosine similarity between L2-normalised TF-IDF query vector and corpus matrix.

#### Step 6 — Context Formatting
```python
build_rag_context(query, index) → (context_text, sources)
```
Formats retrieved chunks as numbered references:
```
## Retrieved Clinical Knowledge Base
[Reference 1 | Source: keratoconus_clinical_guide.txt | Relevance: 0.14]
<chunk text>
---
[Reference 2 | ...]
```

---

## 10. LLM Report Generation

**Files:** `utils/gemini_client.py`, `components/llm_tab.py`  
**Model:** Gemini 2.0 Flash (`gemini-2.0-flash`)

### 10.1 Prompt Structure

```
[System role: specialist ophthalmic AI assistant]

## AI System Analysis Summary
- Ensemble Diagnosis, KC Probability, Severity, Confidence
- Image Model vs Tabular Model breakdown
- Top contributing modalities (from attention weights)
- Top discriminative clinical features (from feature importance)

## Evidence Base (Retrieved from Clinical Knowledge Base)
[RAG-retrieved passages — numbered references]

## Report Sections Requested:
1. Clinical Interpretation
2. Prediction Reasoning
3. Evidence-Based Context
4. Suggested Management
5. Monitoring Recommendations
6. Patient-Friendly Summary
7. Important Caveats
```

### 10.2 Generation Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.3 (low — factual, consistent) |
| Max output tokens | 2000 |
| Model | `gemini-2.0-flash` |

### 10.3 Report Sections

| Section | Content |
|---------|---------|
| **1. Clinical Interpretation** | Specific modality/feature findings with clinical meaning |
| **2. Prediction Reasoning** | Agreement/disagreement between image and tabular streams; confidence interpretation |
| **3. Evidence-Based Context** | Cites retrieved literature; places case in classification systems (Amsler-Krumeich, BAD-D) |
| **4. Suggested Management** | Severity-matched treatment: observation / CXL / contact lens / surgical |
| **5. Monitoring Recommendations** | Follow-up intervals; specific metrics (Kmax, pachymetry, posterior elevation) |
| **6. Patient-Friendly Summary** | 2–3 plain-language sentences |
| **7. Important Caveats** | AI disclaimer; must be reviewed by ophthalmologist |

### 10.4 RAG Toggle

The user can enable/disable RAG per generation via a checkbox. When disabled, the LLM still generates a report but without retrieved evidence — relying entirely on its training knowledge.

---

## 11. UI Tabs & Workflow

### Standard User Workflow

```
1. Sidebar → Load Models
2. Tab 1: Upload Data → upload images and/or CSV → Run Analysis
3. Tab 2: Predictions → view ensemble result, gauge, contribution charts
4. Tab 3: Explainability → Grad-CAM heatmaps, modality attention, feature importance
5. Tab 4: LLM Explanation → Generate RAG-augmented clinical report → Download
```

### Tab 1 — Upload Data (`components/upload_tab.py`)

- 7 individual file uploaders (one per modality), arranged in 2 rows (4+3)
- Real-time image preview on upload
- Progress bar showing how many modalities have been uploaded
- Warning banner listing zero-filled modalities
- CSV uploader with validation: column check, null imputation, transposed feature preview
- **Run Analysis** button (enabled only when models loaded + ≥1 input)
- Progress bar during inference: Image model (10%) → Tabular model (40%) → Ensemble (60%) → Grad-CAM (75%) → Done (100%)

### Tab 2 — Predictions (`components/predictions_tab.py`)

- Top banner: diagnosis badge + KC probability (large font) + severity pill
- 4 metric cards: Normal Prob / KC Prob / Confidence / Severity
- Plotly gauge chart (0–100% dial, colour-coded by severity)
- Ensemble contribution bar chart (image vs tabular contribution)
- Per-model cards: individual predictions from image and tabular models
- Stacked bar chart: Normal vs KC probability for all 3 models side-by-side

### Tab 3 — Explainability (`components/explainability_tab.py`)

Three sub-tabs:

| Sub-tab | Content |
|---------|---------|
| Grad-CAM Heatmaps | JET-coloured overlays per uploaded modality, 4-per-row grid, colour scale legend |
| Modality Importance | Bar chart of cross-modal attention weights; top-modality callout card |
| Feature Importance | Horizontal bar chart of top-20 clinical features; top-5 ranked table with inline bars |

### Tab 4 — LLM Explanation (`components/llm_tab.py`)

- Summary header (diagnosis / KC prob / severity)
- RAG status chip (✅ Active with doc/chunk counts, or ⚠️ unavailable)
- RAG toggle checkbox
- Generate / Regenerate buttons
- Collapsible "Retrieved Knowledge Base References" panel with relevance score bars per passage
- Full markdown clinical report rendered as styled HTML
- Download Report button (`.md` file)
- Disclaimer banner

### Sidebar (`components/sidebar.py`)

- Device indicator (CPU / CUDA / MPS)
- Load Models / Reset buttons
- Model status indicators
- Ensemble weight slider (image ↔ tabular, 0–100%)
- Gemini API key input (password field, session-only)
- About expander

---

## 12. Output Results Reference

### Severity Thresholds

| KC Probability | Severity | Recommended Action |
|---------------|----------|--------------------|
| < 40% | Unlikely | Annual monitoring if risk factors |
| 40–55% | Borderline | 6-monthly topography; FFKC surveillance |
| 55–70% | Mild | CL fitting; CXL if progressive |
| 70–85% | Moderate | Scleral/RGP lenses; CXL strongly indicated |
| > 85% | Severe | Scleral lenses; transplant candidacy assessment |

### Modality Attention Weight Interpretation

| High-weight Modality | Clinical Implication |
|---------------------|---------------------|
| `Elv_P` | Subclinical / early KC — posterior protrusion |
| `Sag_A` + `CT_A` | Established structural KC — curvature + thinning |
| `EC_A` | Epithelial compensation — early focal thinning |
| `Elv_A` | Anterior cone formation — moderate KC |

### Confidence Interpretation

| Confidence | Meaning |
|-----------|---------|
| > 90% | High agreement between streams; reliable prediction |
| 70–90% | Moderate confidence; consider both model outputs |
| 50–70% | Borderline; clinical correlation strongly recommended |
| < 50% | Indeterminate; insufficient or conflicting data |

---

## 13. File Structure

```
keratoconusapp2/
├── app.py                          # Streamlit entry point
├── requirements.txt                # Python dependencies
├── model1.pth                      # Image model checkpoint (EfficientNet-B0 + attention)
├── model2.pkl                      # Tabular model checkpoint (XGBoost/sklearn)
│
├── RAG Files/                      # Clinical knowledge base
│   ├── keratoconus_clinical_guide.txt
│   ├── keratoconus_imaging_modalities.txt
│   └── keratoconus_management_protocols.txt
│
├── utils/
│   ├── model_loader.py             # Model architecture + auto-discovery + loading
│   ├── preprocessor.py             # Image transforms + CSV validation
│   ├── inference.py                # Image/tabular inference + ensemble + Grad-CAM
│   ├── gemini_client.py            # Gemini API call + prompt construction
│   ├── rag_engine.py               # RAG: loading, chunking, TF-IDF, retrieval
│   └── ui_components.py            # CSS loader + layout helpers
│
├── components/
│   ├── sidebar.py                  # Sidebar: models, weights, API key
│   ├── upload_tab.py               # Tab 1: upload + run analysis
│   ├── predictions_tab.py          # Tab 2: ensemble results + charts
│   ├── explainability_tab.py       # Tab 3: Grad-CAM + importance
│   └── llm_tab.py                  # Tab 4: RAG + Gemini report
│
└── assets/
    └── style.css                   # Glassmorphism dark theme CSS
```

---

## 14. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.32.0 | Web UI framework |
| `torch` | ≥2.1.0 | Image model inference + Grad-CAM |
| `torchvision` | ≥0.16.0 | EfficientNet-B0, transforms |
| `scikit-learn` | ≥1.4.0 | Tabular model loading + TF-IDF RAG index |
| `xgboost` | ≥2.0.0 | Tabular model (if checkpoint uses XGBoost) |
| `pandas` | ≥2.1.0 | CSV handling |
| `numpy` | ≥1.26.0 | Numerical operations |
| `pillow` | ≥10.2.0 | Image loading |
| `opencv-python-headless` | ≥4.9.0 | Grad-CAM overlay |
| `matplotlib` | ≥3.8.0 | Colormaps |
| `plotly` | ≥5.18.0 | Interactive charts |
| `google-generativeai` | ≥0.5.0 | Gemini API client |
| `captum` | ≥0.7.0 | Optional advanced XAI |
| `joblib` | ≥1.3.0 | Tabular model serialisation |
| `pypdf` | ≥4.0.0 | PDF parsing for RAG Files |

---

## Important Disclaimer

> ⚠️ **KeratoScan AI is a research and educational tool only.**  
> It is **not** a certified medical device and must **not** be used for clinical diagnosis.  
> All outputs must be reviewed and independently verified by a qualified ophthalmologist before any clinical decision is made.  
> The AI system may produce errors, hallucinations, or incomplete assessments.
