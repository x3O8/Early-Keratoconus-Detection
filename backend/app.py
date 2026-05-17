import io
import os
import sys
import traceback
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import inference_service
import model_service
from mm_architecture import MODALITIES
from mm_inference import gradcam_overlays_b64
from preprocessor import validate_csv

app = Flask(__name__)
CORS(app)

cache_holder = model_service.cache


@app.before_request
def _ensure_models() -> None:
    if cache_holder.image_model is None or cache_holder.tabular_model is None:
        cache_holder.refresh()


@app.get("/health")
def health() -> Any:
    if cache_holder.image_model is None or cache_holder.tabular_model is None:
        cache_holder.refresh()
    return jsonify(
        {
            "status": "ok",
            "device": str(cache_holder.device),
            "modalities": MODALITIES,
            "image_model": {
                "loaded": cache_holder.image_model is not None,
                "path": cache_holder.image_path,
                "meta": cache_holder.image_meta,
            },
            "tabular_model": {
                "loaded": cache_holder.tabular_model is not None,
                "path": cache_holder.tabular_path,
                "meta": cache_holder.tabular_meta,
            },
            "errors": cache_holder._errors,
        }
    )


def _modalities_list():
    from mm_architecture import MODALITIES

    return list(MODALITIES)


@app.post("/predict")
def predict() -> Any:
    try:
        tensors, pils_unused, parse_warnings = inference_service.parse_modalities_from_request(request)

        df = None
        csv_warnings: list[str] = []
        csv_errors: list[str] = []
        if "tabular" in request.files and request.files["tabular"].filename:
            raw_csv = request.files["tabular"].read()
            df, csv_errors, csv_warnings = validate_csv(io.BytesIO(raw_csv), cache_holder.tabular_model)
            if csv_errors:
                return jsonify({"error": "CSV validation failed", "details": csv_errors, "warnings": csv_warnings}), 400

        if not tensors and df is None:
            return (
                jsonify(
                    {
                        "error": "Provide at least one of: modality images (fields CT_A … Sag_P) and/or a tabular CSV file 'tabular'.",
                        "modalities": MODALITIES,
                    }
                ),
                400,
            )

        w_img = float(request.form.get("w_image", request.form.get("w1", 0.65)))
        w_tab = float(request.form.get("w_tabular", request.form.get("w2", 0.35)))

        out = inference_service.run_full_analysis(tensors, df, w_img, w_tab, cache_holder)
        out["warnings"] = parse_warnings + csv_warnings
        return jsonify(out)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/gradcam")
def gradcam_route() -> Any:
    try:
        tensors, pils, _w = inference_service.parse_modalities_from_request(request)
        if not tensors:
            return (
                jsonify(
                    {
                        "error": "Upload at least one modality image (same field names as Streamlit: CT_A, EC_A, …).",
                        "modalities": MODALITIES,
                    }
                ),
                400,
            )

        if cache_holder.image_model is None:
            return jsonify({"error": "Image model not loaded"}), 500

        target_class = int(request.form.get("target_class", 1))
        alpha = float(request.form.get("alpha", 0.45))
        only = request.form.get("only_modality") or None
        if only and only not in MODALITIES:
            only = None

        overlays = gradcam_overlays_b64(
            cache_holder.image_model,
            tensors,
            pils,
            cache_holder.device,
            target_class=target_class,
            alpha=alpha,
            only_modality=only,
        )
        return jsonify(
            {
                "kind": "multimodal",
                "target_class": target_class,
                "only_modality": only,
                "by_modality": overlays.get("by_modality", {}),
                "integrated_gradients": None,
                "attention_rollout": None,
                "note": "Integrated gradients are disabled for the 7-encoder graph; use per-modality Grad-CAM.",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    cache_holder.refresh()
    app.run(host="127.0.0.1", port=5000, debug=False)
