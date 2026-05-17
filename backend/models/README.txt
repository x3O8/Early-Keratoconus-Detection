Place model weights under `backend/models/`:

## Multimodal image model (7 × ResNet-18 encoders + transformer fusion)

- Any `*.pth` / `*.pt` whose state dict contains keys like `encoders.0...` will be loaded into **MultiModalKeratoconusModel** (same as `utils/model_loader.py` in the Streamlit app).
- The loader picks the checkpoint with the **best key overlap** with that architecture.
- If nothing matches, an **untrained** multimodal model is used so the UI still runs.

## Tabular model

- Preferred filenames: `tabular.pkl`, `model2.pkl`, `clinical.pkl`, or any `*.pkl` / `*.joblib` in this folder (sorted with tabular/model2 names first).
- Must support **`predict_proba`** or **`decision_function`** (sklearn-style).
- If no file is found, a small **demo LogisticRegression** trained on synthetic data is used; CSV columns must be **`feature_0` … `feature_9`** for that demo.

## API upload field names (must match Streamlit modalities)

`CT_A`, `EC_A`, `EC_P`, `Elv_A`, `Elv_P`, `Sag_A`, `Sag_P` — optional each; missing ones are **zero-filled** on the server.

**CSV** multipart field name: **`tabular`** (one header row + one patient row).
