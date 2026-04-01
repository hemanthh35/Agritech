# AgriTech Precision — AI Agriculture Web Application

End-to-end **precision agriculture** demo: **crop** and **fertilizer** recommendation with locally trained **Random Forest** models, plus **plant disease** screening using a **pre-trained EfficientNetB0** (Keras) model. The stack is **Flask** (API + server-rendered UI), **HTML/CSS/JS**, and **scikit-learn** / **TensorFlow**.

---

## Table of contents

1. [Features](#features)
2. [Project structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Training the ML models](#training-the-ml-models)
6. [Disease model (Colab export)](#disease-model-colab-export)
7. [Running the application](#running-the-application)
8. [Web pages](#web-pages)
9. [HTTP API](#http-api)
10. [Data & models (reference)](#data--models-reference)
11. [Metrics & evaluation](#metrics--evaluation)
12. [Troubleshooting](#troubleshooting)
13. [Production notes](#production-notes)

---

## Features

| Module | Method | Description |
|--------|--------|-------------|
| **Crop recommendation** | `RandomForestClassifier` (local train) | Predicts crop from N, P, K, temperature, humidity, pH, rainfall. |
| **Fertilizer recommendation** | `RandomForestClassifier` (local train) | Predicts fertilizer from weather, moisture, soil pH, soil/crop type, NPK; returns **confidence** and a short **explanation**. |
| **Disease detection** | **EfficientNetB0** Keras model (inference only) | Leaf image upload → class label + confidence. |

Synthetic datasets are **3,000 rows** each (see below). The fertilizer dataset is **balanced** (375 samples per fertilizer class).

---

## Project structure

```text
agritech/
├── app.py                 # Flask app: UI routes + JSON API
├── requirements.txt
├── train_crop.py          # Train & save crop model + metrics_crop.json
├── train_fertilizer.py    # Train & save fertilizer model + fertilizer_metrics.json
├── training_metrics.py    # Shared helpers for crop metrics JSON
├── export_labels_colab.py # Snippet/comments to export class_labels from Colab
├── traning.ipynb          # Reference: EfficientNet training in Colab
├── dataset/               # Generated CSVs (after training)
│   ├── crop_data.csv
│   └── fertilizer_data.csv
├── models/                # Artifacts (you add disease model files)
│   ├── crop_model.pkl
│   ├── fertilizer_model.pkl
│   ├── metrics_crop.json
│   ├── fertilizer_metrics.json
│   ├── class_labels.json  # Disease class names (order = model output index)
│   ├── best_model.keras   # Optional: your Colab export (preferred)
│   └── final_model.h5     # Optional: alternative Keras/H5 export
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── images/            # UI assets (photos, SVGs)
└── templates/             # Jinja2 HTML
    ├── base.html
    ├── index.html
    ├── crop.html
    ├── fertilizer.html
    ├── disease.html
    └── metrics.html
```

---

## Requirements

- **Python 3.10+** recommended (typing and ecosystem alignment with TensorFlow 2.15+).
- **pip** (or conda) for dependencies listed in `requirements.txt`.
- For **disease inference**: a compatible **TensorFlow** build for your OS (CPU/GPU).

---

## Installation

From the project root (`agritech/`):

```bash
cd path/to/agritech
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Training the ML models

Run **after** installing dependencies. Scripts write CSVs under `dataset/` and artifacts under `models/`.

### Crop model

```bash
python train_crop.py
```

- **Output:** `models/crop_model.pkl`, `dataset/crop_data.csv`, `models/metrics_crop.json`
- **Target crops (10):** rice, wheat, maize, cotton, sugarcane, chickpea, kidneybeans, mungbean, blackgram, lentil  
- **Features:** `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`  
- **Algorithm:** `RandomForestClassifier` (200 trees, stratified 80/20 split)

### Fertilizer model

```bash
python train_fertilizer.py
```

- **Output:** `models/fertilizer_model.pkl`, `dataset/fertilizer_data.csv`, `models/fertilizer_metrics.json`
- **Classes (8, 375 rows each):** DAP, MOP, NPK_10_26_26, NPK_19_19_19, Organic_Compost, SSP, Urea, Zinc_Sulphate  
- **Features:** `temperature`, `humidity`, `moisture`, `soil_pH`, `soil_type`, `crop_type`, `N`, `P`, `K`  
- **Algorithm:** `RandomForestClassifier` (`n_estimators=200`, `class_weight='balanced'`), `LabelEncoder` on categoricals  

Re-training **overwrites** the corresponding `.pkl` and metrics JSON files.

---

## Disease model (Colab export)

Do **not** retrain the disease model inside this repo unless you intentionally replace it.

1. Export from Colab (e.g. `best_model.keras` or `final_model.h5`) as built in `traning.ipynb` (EfficientNetB0, 224×224, `efficientnet.preprocess_input`).
2. Copy into `models/`:
   - Prefer **`models/best_model.keras`**, or  
   - **`models/final_model.h5`** if the app should load H5 instead.

3. **Class names:** `models/class_labels.json` must list **one string per output neuron**, where index `i` matches softmax output `i` (same order as Keras `flow_from_directory` class indices).  
   - Default file ships with a **PlantVillage-style** alphabetical list (38 classes).  
   - If your training folders differ, regenerate labels in Colab and replace the file. See comments in `export_labels_colab.py`.

If no disease model file is present, **crop** and **fertilizer** routes still work; **`/predict_disease`** returns an error JSON explaining the missing file.

---

## Running the application

```bash
python app.py
```

Default URL: **http://127.0.0.1:5000** (binds to `0.0.0.0` as configured).

Stop with `Ctrl+C` in the terminal.

---

## Web pages

| Path | Purpose |
|------|---------|
| `/` | Landing page with links to the three tools |
| `/crop` | Crop recommendation form |
| `/fertilizer` | Fertilizer form (validation, confidence, explanation) |
| `/disease` | Leaf image upload |
| `/metrics` | Training metrics UI (loads `/api/metrics`) |

Static assets live under `static/`; templates under `templates/`.

---

## HTTP API

All JSON routes expect **`Content-Type: application/json`** unless noted.

### `POST /predict_crop`

**Body (example):**

```json
{
  "N": 90,
  "P": 45,
  "K": 45,
  "temperature": 25,
  "humidity": 70,
  "ph": 6.5,
  "rainfall": 100
}
```

**Response:**

```json
{ "crop": "maize" }
```

### `POST /predict_fertilizer`

**Body (example):**

```json
{
  "temperature": 26,
  "humidity": 68,
  "moisture": 45,
  "soil_pH": 6.5,
  "soil_type": "Loamy",
  "crop_type": "wheat",
  "N": 50,
  "P": 45,
  "K": 40
}
```

**Response:**

```json
{
  "prediction": "NPK_19_19_19",
  "confidence": "88.0%",
  "input_summary": { ... },
  "explanation": "Based on ..."
}
```

Valid `soil_type` values: Sandy, Loamy, Black, Red, Clayey.  
Valid `crop_type` values: rice, wheat, maize, cotton, sugarcane, pulses.

### `POST /predict_disease`

- **`multipart/form-data`** with file field **`image`** (leaf photo).
- **Response:** `disease`, `confidence` (float 0–1), `class_index`.

### `GET /api/metrics`

Returns `crop` and `fertilizer` metric objects (from JSON files) and `_meta` with metric **file names** (`crop_file`, `fertilizer_file`).

### Errors

- **4xx/5xx** JSON bodies typically include `{ "error": "message" }`.
- **413** if upload exceeds Flask `MAX_CONTENT_LENGTH` (16 MB).

---

## Data & models (reference)

| Artifact | Description |
|----------|-------------|
| `crop_model.pkl` | `joblib` dict: `model`, `label_encoder`, `feature_columns` |
| `fertilizer_model.pkl` | `joblib` dict: `model`, `label_encoder_y`, `categorical_encoders`, `numeric_features`, `categorical_features`, `feature_columns` |
| `crop_data.csv` / `fertilizer_data.csv` | Regenerated each train run |

Disease inference uses the same preprocessing as training: RGB, resize **224×224**, **`tensorflow.keras.applications.efficientnet.preprocess_input`**.

---

## Metrics & evaluation

- **Crop:** `models/metrics_crop.json` (from `training_metrics.build_classification_metrics`).
- **Fertilizer:** `models/fertilizer_metrics.json` (includes `classification_report_text`, confusion matrix, hyperparameters summary).

The **`/metrics`** page reads **`GET /api/metrics`** and can **refresh** without reloading the whole app.  
Legacy filename **`metrics_fertilizer.json`** is still read by the API if `fertilizer_metrics.json` is absent.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `Missing ... crop_model.pkl` / `fertilizer_model.pkl` | Run `train_crop.py` / `train_fertilizer.py`. |
| Disease endpoint 503 | Add `best_model.keras` or `final_model.h5` under `models/`. |
| Wrong disease names | Replace `models/class_labels.json` with labels exported from your Colab dataset (see `export_labels_colab.py`). |
| TensorFlow slow first call | First `predict_disease` loads the Keras model (lazy load). |
| Port 5000 in use | `set PORT=8000` (Windows) or `export PORT=8000` (Unix) before `python app.py`, or change the port in `app.py`. |
| Windows console Unicode errors in training scripts | Training scripts use ASCII arrows in prints; avoid non-CP1252 symbols in your own edits. |

---

## Production notes

- The built-in Flask server is **not** meant for production. Use a WSGI server (e.g. **Gunicorn** on Linux, **Waitress** on Windows) behind a reverse proxy, set `debug=False` (already off), configure secrets and HTTPS, and tighten upload limits as needed.
- Pin exact dependency versions in `requirements.txt` for reproducible deploys.
- Keep **large** files (`*.pkl`, `*.keras`, `*.h5`, images) out of version control if the repo size matters; document where to copy them (this README).

---

## License & credits

- Application code: your project / client license as applicable.
- Stock **photos** under `static/images/` may originate from **Unsplash**; retain upstream license terms if you redistribute.
- **PlantVillage**-style naming in `class_labels.json` is for compatibility with common public datasets; verify alignment with your own trained model.

---

## Quick start checklist

1. `pip install -r requirements.txt`  
2. `python train_crop.py`  
3. `python train_fertilizer.py`  
4. Copy `best_model.keras` (and optional custom `class_labels.json`) into `models/`  
5. `python app.py` → open http://127.0.0.1:5000  

For questions or changes, edit the training scripts and templates in this repository and re-run training as needed.
