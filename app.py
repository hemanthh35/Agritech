"""
Flask API + web UI for precision agriculture: crop, fertilizer, and disease prediction.
Production-ready with lazy loading, error handling, and dual deployment support.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import joblib

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

# Limit upload size (16 MB)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

_crop_bundle = None
_fertilizer_bundle = None
_disease_model = None
_class_labels: list[str] = []


def _load_class_labels() -> list[str]:
    path = MODEL_DIR / "class_labels.json"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data) if isinstance(data, list) else []


def get_crop_bundle():
    global _crop_bundle
    if _crop_bundle is None:
        path = MODEL_DIR / "crop_model.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run train_crop.py")
        _crop_bundle = joblib.load(path)
    return _crop_bundle


def get_fertilizer_bundle():
    global _fertilizer_bundle
    if _fertilizer_bundle is None:
        path = MODEL_DIR / "fertilizer_model.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run train_fertilizer.py")
        _fertilizer_bundle = joblib.load(path)
    return _fertilizer_bundle


def get_disease_model():
    """Lazy-load TensorFlow/Keras model (heavy). Handles Keras 2/3 compatibility."""
    global _disease_model, _class_labels
    if _disease_model is None:
        import tensorflow as tf

        h5_path = MODEL_DIR / "final_model.h5"
        keras_path = MODEL_DIR / "best_model.keras"
        
        # Prefer .h5 (Keras 2.x compatible), skip .keras (Keras 3.x incompatible)
        if h5_path.is_file():
            try:
                logger.info(f"Loading disease model from {h5_path}")
                _disease_model = tf.keras.models.load_model(h5_path)
                logger.info("✓ Disease model loaded successfully")
                _class_labels = _load_class_labels()
                return _disease_model
            except Exception as e:
                logger.error(f"Failed to load h5 model: {e}")
        
        # Fallback: try .keras (may fail if Keras 3.x format on Keras 2.x system)
        if keras_path.is_file():
            logger.warning(
                f"Using .keras file ({keras_path}); this may fail if saved with Keras 3.x. "
                f"For production: convert to .h5 format on Google Colab using colab_convert_model.py"
            )
            try:
                _disease_model = tf.keras.models.load_model(keras_path)
                _class_labels = _load_class_labels()
                return _disease_model
            except Exception as e:
                logger.error(f"Failed to load .keras file: {str(e)[:200]}")
        
        # No model found
        raise FileNotFoundError(
            "Disease model missing. Expected: models/final_model.h5 "
            "(or models/best_model.keras converted from Colab). "
            "See colab_convert_model.py for conversion instructions."
        )


def _fertilizer_explanation(prediction: str, summary: dict) -> str:
    """Short rule-based rationale from input summary (complements model output)."""
    try:
        n = float(summary.get("N", 0))
        p = float(summary.get("P", 0))
        k = float(summary.get("K", 0))
        ph = float(summary.get("soil_pH", 7))
        soil = str(summary.get("soil_type", ""))
        moisture = float(summary.get("moisture", 50))
    except (TypeError, ValueError):
        return f"The model suggests {prediction} based on your soil, crop, and nutrient profile."

    parts = []
    if n < 40:
        parts.append("nitrogen appears limited")
    elif n > 70:
        parts.append("nitrogen levels are relatively high")
    if p < 35:
        parts.append("phosphorus looks low")
    elif p > 55:
        parts.append("phosphorus is adequate to high")
    if k < 38:
        parts.append("potassium may need topping up")
    if ph > 7.3:
        parts.append("alkaline pH")
    elif ph < 6.0:
        parts.append("acidic soil")
    if moisture < 32:
        parts.append("dry soil moisture")
    elif moisture > 58:
        parts.append("high moisture")
    if soil:
        parts.append(f"{soil.lower()} soil")

    ctx = ", ".join(parts) if parts else "your combined readings"
    return (
        f"Based on {ctx}, {prediction} is the closest match among the trained fertilizer classes."
    )


def preprocess_disease_image(pil_image: Image.Image) -> np.ndarray:
    """Resize to 224x224 RGB and apply EfficientNetB0 preprocess_input (matches training)."""
    from tensorflow.keras.applications.efficientnet import preprocess_input

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(pil_image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/crop")
def crop_page():
    return render_template("crop.html")


@app.route("/fertilizer")
def fertilizer_page():
    return render_template("fertilizer.html")


@app.route("/disease")
def disease_page():
    return render_template("disease.html")


@app.route("/metrics")
def metrics_page():
    metrics_data: dict = {"crop": None, "fertilizer": None, "_meta": {}}

    crop_path = MODEL_DIR / "metrics_crop.json"
    if crop_path.is_file():
        with open(crop_path, encoding="utf-8") as f:
            metrics_data["crop"] = json.load(f)
        metrics_data["_meta"]["crop_file"] = crop_path.name

    fert_path = MODEL_DIR / "fertilizer_metrics.json"
    fert_name = "fertilizer_metrics.json"
    if not fert_path.is_file():
        fert_path = MODEL_DIR / "metrics_fertilizer.json"
        fert_name = "metrics_fertilizer.json"
    if fert_path.is_file():
        with open(fert_path, encoding="utf-8") as f:
            metrics_data["fertilizer"] = json.load(f)
        metrics_data["_meta"]["fertilizer_file"] = fert_name

    return render_template("metrics.html", metrics_data=metrics_data)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """JSON metrics from last local training."""
    out: dict = {"crop": None, "fertilizer": None, "_meta": {}}

    crop_path = MODEL_DIR / "metrics_crop.json"
    if crop_path.is_file():
        with open(crop_path, encoding="utf-8") as f:
            out["crop"] = json.load(f)
        out["_meta"]["crop_file"] = crop_path.name

    fert_path = MODEL_DIR / "fertilizer_metrics.json"
    fert_name = "fertilizer_metrics.json"
    if not fert_path.is_file():
        fert_path = MODEL_DIR / "metrics_fertilizer.json"
        fert_name = "metrics_fertilizer.json"
    if fert_path.is_file():
        with open(fert_path, encoding="utf-8") as f:
            out["fertilizer"] = json.load(f)
        out["_meta"]["fertilizer_file"] = fert_name

    return jsonify(out)


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400

        bundle = get_crop_bundle()
        cols = bundle["feature_columns"]
        missing = [c for c in cols if c not in payload]
        if missing:
            return jsonify({"error": f"Missing keys: {missing}"}), 400

        X = np.array([[float(payload[c]) for c in cols]], dtype=np.float32)
        pred = bundle["model"].predict(X)[0]
        label = bundle["label_encoder"].inverse_transform([int(pred)])[0]
        return jsonify({"crop": str(label)})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400

        bundle = get_fertilizer_bundle()
        num_cols = bundle["numeric_features"]
        cat_cols = bundle["categorical_features"]
        encoders = bundle["categorical_encoders"]

        needed = num_cols + cat_cols
        missing = [c for c in needed if c not in payload]
        if missing:
            return jsonify({"error": f"Missing keys: {missing}"}), 400

        input_summary: dict = {}
        for c in needed:
            v = payload[c]
            if isinstance(v, (int, float)):
                input_summary[c] = float(v)
            else:
                input_summary[c] = v

        num = np.array([[float(payload[c]) for c in num_cols]], dtype=np.float64)
        cat_blocks = []
        for col in cat_cols:
            val = str(payload[col]).strip()
            le = encoders[col]
            if val not in le.classes_:
                return jsonify({"error": f"Unknown {col}: {val!r}"}), 400
            cat_blocks.append(le.transform([val]).reshape(-1, 1))
        X = np.hstack([num] + cat_blocks)

        clf = bundle["model"]
        pred_idx = int(clf.predict(X)[0])
        probs = clf.predict_proba(X)[0]
        conf = float(np.clip(probs[pred_idx], 0.0, 1.0))
        fert = bundle["label_encoder_y"].inverse_transform([pred_idx])[0]
        prediction = str(fert)
        conf_pct = f"{conf * 100:.1f}%"

        explanation = _fertilizer_explanation(prediction, input_summary)

        return jsonify(
            {
                "prediction": prediction,
                "confidence": conf_pct,
                "input_summary": input_summary,
                "explanation": explanation,
            }
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Missing file field 'image'"}), 400
        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        pil = Image.open(file.stream)
        model = get_disease_model()
        x = preprocess_disease_image(pil)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        labels = _class_labels or _load_class_labels()
        if idx < len(labels):
            name = labels[idx]
        else:
            name = f"class_{idx}"

        return jsonify({"disease": name, "confidence": confidence, "class_index": idx})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except OSError as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400
    except Exception as e:  # pragma: no cover
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File too large (max 16 MB)"}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 error: {e}")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


if __name__ == "__main__":
    # Preload class labels for disease route metadata
    _class_labels.extend(_load_class_labels())
    
    # Get port from environment or default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    
    # Production safety: never use debug=True in production
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting AgriTech API on port {port} (debug={debug_mode})")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
