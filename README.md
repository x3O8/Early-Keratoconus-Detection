# Early Keratoconus Detection — Localhost AI Dashboard

A **lightweight localhost** stack for research demos: **React (Vite + Tailwind + Framer Motion)** frontend and a **small Flask** backend with the same **7-modality multimodal PyTorch model** and **sklearn tabular** path as the Streamlit app, **weighted fusion**, **per-modality Grad-CAM**, and **clinical-style explanations**.

> **Not for clinical use.** Research and presentation only.

---

## Project layout

| Path | Role |
|------|------|
| `frontend/` | Vite + React + Tailwind UI (proxies `/api` → Flask) |
| `backend/` | Flask app, model loading, explainability |
| `backend/models/` | Multimodal `.pth` + tabular `.pkl` (see `backend/models/README.txt`) |
| `examples/` | Example JSON shapes for API responses |

The repository still contains the **Streamlit** KeratoScan app (`app.py` at repo root, `utils/`, `components/`). The **Flask + React** dashboard mirrors its **7-modality** inputs, **tabular CSV**, **fusion**, and **per-modality Grad-CAM** behaviour.

---

## 1. Backend (Flask)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

API (default `http://127.0.0.1:5000`):

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Device, modality list, multimodal + tabular load metadata |
| `POST` | `/predict` | Multipart: optional **`CT_A`…`Sag_P`**, optional **`tabular`** CSV, **`w1`/`w2`** (image vs tabular fusion) |
| `POST` | `/gradcam` | Same modality fields + **`target_class`**, **`alpha`** → JSON **`by_modality`** |

- **Image:** `MultiModalKeratoconusModel` (same architecture as `utils/model_loader.py`); missing modalities are **zero-filled** (`utils/inference.run_image_inference`).
- **Tabular:** sklearn-like `*.pkl` / `*.joblib` in `backend/models/`; CSV rules match `utils/preprocessor.validate_csv`.
- **Fusion:** same as Streamlit `ensemble_fusion`; if only one branch has real data, the other uses **50/50** placeholder probabilities (see `components/upload_tab.py`).

**GPU:** CUDA or MPS when available; otherwise CPU.

---

## 2. Frontend (Vite)

```powershell
cd frontend
npm install
npm run dev
```

Open the printed localhost URL (usually `http://127.0.0.1:5173`). The dev server proxies API calls to `http://127.0.0.1:5000` via the `/api` prefix.

For a static build served elsewhere, set:

```text
VITE_API_BASE=http://127.0.0.1:5000
```

---

## 3. Fusion

Server-side:

\[
\text{fused} = \frac{w_\text{img}}{w_\text{img}+w_\text{tab}}\, p^{(\text{image})} + \frac{w_\text{tab}}{w_\text{img}+w_\text{tab}}\, p^{(\text{tabular})}
\]

Form fields `w1` / `w2` map to image and tabular weights (same defaults as Streamlit: 0.65 / 0.35).

---

## 4. Explainability

- **GradCAM:** per modality, last conv of each encoder (`utils/inference.compute_gradcam`). API returns `by_modality[CT_A].overlay_b64`, etc.
- **Integrated gradients / ViT rollout:** not implemented for this multimodal graph (use the Streamlit app’s other tabs if you need LLM text; Grad-CAM parity is the focus here).

---

## 5. Example API payloads

See `examples/mock_predict_response.json` for a representative `/predict` JSON shape.

---

## Disclaimer

Outputs are **not** validated medical devices. All decisions require a qualified clinician and appropriate diagnostics (e.g., topography, tomography).
