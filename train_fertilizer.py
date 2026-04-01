"""
Train RandomForest fertilizer recommender: 3000 rows, perfectly balanced 8 classes.
Saves models/fertilizer_model.pkl and models/fertilizer_metrics.json.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
CSV_PATH = DATA_DIR / "fertilizer_data.csv"
MODEL_PATH = MODEL_DIR / "fertilizer_model.pkl"
METRICS_PATH = MODEL_DIR / "fertilizer_metrics.json"

SOIL_TYPES = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
CROP_TYPES = ["rice", "wheat", "maize", "cotton", "sugarcane", "pulses"]

# Exact list: equal samples per class (3000 / 8 = 375)
FERTILIZERS = [
    "DAP",
    "MOP",
    "NPK_10_26_26",
    "NPK_19_19_19",
    "Organic_Compost",
    "SSP",
    "Urea",
    "Zinc_Sulphate",
]

# Separable centroids per fertilizer (N, P, K, temp, humidity, moisture, soil_pH)
CENTROIDS: dict[str, tuple[float, float, float, float, float, float, float]] = {
    "Urea": (32.0, 52.0, 46.0, 28.5, 76.0, 56.0, 6.35),
    "DAP": (58.0, 24.0, 44.0, 22.0, 64.0, 36.0, 6.85),
    "MOP": (48.0, 48.0, 28.0, 27.0, 70.0, 48.0, 6.55),
    "NPK_10_26_26": (44.0, 38.0, 36.0, 25.0, 68.0, 44.0, 6.15),
    "NPK_19_19_19": (52.0, 42.0, 40.0, 26.5, 72.0, 50.0, 6.45),
    "Organic_Compost": (40.0, 46.0, 42.0, 24.0, 62.0, 42.0, 6.25),
    "SSP": (50.0, 20.0, 48.0, 23.0, 60.0, 40.0, 7.05),
    "Zinc_Sulphate": (46.0, 50.0, 50.0, 30.0, 66.0, 46.0, 7.65),
}

NUMERIC_FEATURES = ["temperature", "humidity", "moisture", "N", "P", "K", "soil_pH"]
CAT_FEATURES = ["soil_type", "crop_type"]
TARGET = "fertilizer"
N_ROWS = 3000
RNG = np.random.default_rng(2026)
SIGMA_NUM = np.array([2.2, 4.0, 4.5, 4.5, 3.5, 3.5, 0.22], dtype=np.float64)


def build_synthetic_fertilizer_dataset(n_rows: int = N_ROWS) -> pd.DataFrame:
    assert n_rows % len(FERTILIZERS) == 0
    per = n_rows // len(FERTILIZERS)
    rows: list[dict] = []

    for fert in FERTILIZERS:
        base = np.array(CENTROIDS[fert], dtype=np.float64)
        for _ in range(per):
            nums = base + RNG.normal(0, 1, size=7) * SIGMA_NUM
            temp = float(np.clip(nums[3], 12.0, 40.0))
            hum = float(np.clip(nums[4], 35.0, 92.0))
            moisture = float(np.clip(nums[5], 18.0, 78.0))
            n = float(np.clip(nums[0], 8.0, 115.0))
            p = float(np.clip(nums[1], 10.0, 95.0))
            k = float(np.clip(nums[2], 12.0, 95.0))
            ph = float(np.clip(nums[6], 5.4, 8.2))
            soil = SOIL_TYPES[int(RNG.integers(0, len(SOIL_TYPES)))]
            crop = CROP_TYPES[int(RNG.integers(0, len(CROP_TYPES)))]
            rows.append(
                {
                    "temperature": temp,
                    "humidity": hum,
                    "moisture": moisture,
                    "soil_type": soil,
                    "crop_type": crop,
                    "N": n,
                    "P": p,
                    "K": k,
                    "soil_pH": ph,
                    TARGET: fert,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert len(df) == n_rows
    vc = df[TARGET].value_counts()
    assert (vc == per).all(), vc
    return df


def main() -> None:
    import json

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = build_synthetic_fertilizer_dataset(N_ROWS)
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {CSV_PATH} ({len(df)} rows); per-class count = {N_ROWS // len(FERTILIZERS)}")

    encoders: dict[str, LabelEncoder] = {}
    X_parts = [df[NUMERIC_FEATURES].values.astype(np.float64)]
    for col in CAT_FEATURES:
        le = LabelEncoder()
        X_parts.append(le.fit_transform(df[col].astype(str)).reshape(-1, 1))
        encoders[col] = le

    X = np.hstack(X_parts)
    le_y = LabelEncoder()
    le_y.fit(FERTILIZERS)
    y = le_y.transform(df[TARGET].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=24,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    target_names = le_y.classes_.tolist()
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    acc_train = accuracy_score(y_train, y_train_pred)
    acc = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    p_macro, r_macro, _, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro", zero_division=0
    )
    cm_test = confusion_matrix(
        y_test, y_test_pred, labels=list(range(len(target_names)))
    )
    report_str = classification_report(
        y_test,
        y_test_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
    )
    print(f"Train accuracy: {acc_train:.4f}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1 (macro): {f1_macro:.4f}")
    print("Classification report (test):\n", report_str)

    prec, rec, _, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average=None, zero_division=0, labels=list(range(len(target_names)))
    )
    bad = [
        (target_names[i], float(prec[i]), float(rec[i]))
        for i in range(len(target_names))
        if prec[i] <= 0 or rec[i] <= 0
    ]
    if bad:
        raise RuntimeError(f"Zero precision/recall for: {bad}")

    report_dict = classification_report(
        y_test,
        y_test_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    def json_safe(o):
        if isinstance(o, dict):
            return {str(k): json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    metrics_payload = {
        "model_name": "fertilizer_random_forest",
        "n_rows": N_ROWS,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "classes": FERTILIZERS,
        "samples_per_class": N_ROWS // len(FERTILIZERS),
        "accuracy_train": float(acc_train),
        "accuracy_test": float(acc),
        "f1_macro": float(f1_macro),
        "f1_macro_test": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "classification_report_test": json_safe(report_dict),
        "classification_report_text": report_str,
        "confusion_matrix_test": cm_test.tolist(),
        "confusion_matrix_labels": target_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CAT_FEATURES,
        "hyperparameters": {
            "n_estimators": 200,
            "class_weight": "balanced",
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")

    bundle = {
        "model": clf,
        "label_encoder_y": le_y,
        "categorical_encoders": encoders,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CAT_FEATURES,
        "feature_columns": NUMERIC_FEATURES + CAT_FEATURES,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
