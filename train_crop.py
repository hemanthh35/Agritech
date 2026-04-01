"""
Train RandomForest crop recommender on a synthetic 3000-row dataset.
Saves models/crop_model.pkl with model, label encoder, and feature columns.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from training_metrics import build_classification_metrics, save_metrics_json

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
CSV_PATH = DATA_DIR / "crop_data.csv"
MODEL_PATH = MODEL_DIR / "crop_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics_crop.json"

# Target crops and rough feature centroids (N, P, K, temp, humidity, ph, rainfall)
CROP_PROFILES: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "rice": (80, 48, 40, 24.0, 82.0, 6.2, 205.0),
    "wheat": (95, 52, 38, 21.0, 65.0, 6.8, 95.0),
    "maize": (90, 42, 50, 26.0, 70.0, 6.5, 88.0),
    "cotton": (120, 45, 40, 28.0, 68.0, 7.0, 72.0),
    "sugarcane": (75, 55, 85, 27.0, 75.0, 6.0, 115.0),
    "chickpea": (40, 68, 40, 19.0, 62.0, 7.2, 72.0),
    "kidneybeans": (20, 60, 20, 20.0, 70.0, 6.6, 105.0),
    "mungbean": (22, 52, 24, 28.0, 72.0, 6.8, 92.0),
    "blackgram": (42, 68, 22, 30.0, 65.0, 7.0, 65.0),
    "lentil": (18, 68, 18, 20.0, 64.0, 6.9, 48.0),
}

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
RNG = np.random.default_rng(42)
N_ROWS = 3000


def build_synthetic_crop_dataset(n_rows: int = N_ROWS) -> pd.DataFrame:
    """Generate exactly n_rows with crop-specific noisy features."""
    crops = list(CROP_PROFILES.keys())
    rows: list[dict] = []
    per_crop = n_rows // len(crops)
    remainder = n_rows % len(crops)

    for i, crop in enumerate(crops):
        n_samples = per_crop + (1 if i < remainder else 0)
        n_base, p, k, t, h, ph, rain = CROP_PROFILES[crop]
        for _ in range(n_samples):
            rows.append(
                {
                    "N": max(0.0, n_base + RNG.normal(0, 12)),
                    "P": max(0.0, p + RNG.normal(0, 10)),
                    "K": max(0.0, k + RNG.normal(0, 10)),
                    "temperature": float(np.clip(t + RNG.normal(0, 3), 8, 45)),
                    "humidity": float(np.clip(h + RNG.normal(0, 8), 25, 99)),
                    "ph": float(np.clip(ph + RNG.normal(0, 0.35), 4.5, 8.5)),
                    "rainfall": max(0.0, rain + RNG.normal(0, 35)),
                    "crop": crop,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert len(df) == n_rows, f"Expected {n_rows} rows, got {len(df)}"
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = build_synthetic_crop_dataset(N_ROWS)
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {CSV_PATH} ({len(df)} rows)")

    X = df[FEATURE_COLUMNS].values
    le = LabelEncoder()
    y = le.fit_transform(df["crop"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    target_names = le.classes_.tolist()
    metrics = build_classification_metrics(
        model_name="crop_random_forest",
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        target_names=target_names,
        extras={
            "algorithm": "RandomForestClassifier",
            "n_estimators": clf.n_estimators,
            "test_size": 0.2,
            "random_state": 42,
            "dataset_rows": N_ROWS,
            "features": FEATURE_COLUMNS,
        },
    )
    save_metrics_json(METRICS_PATH, metrics)
    print(f"Train accuracy: {metrics['accuracy_train']:.4f}")
    print(f"Test accuracy:  {metrics['accuracy_test']:.4f}")
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f} | weighted: {metrics['f1_weighted']:.4f}")
    print(f"Saved metrics -> {METRICS_PATH}")

    bundle = {
        "model": clf,
        "label_encoder": le,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
