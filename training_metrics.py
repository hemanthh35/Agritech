"""
Compute and persist classification metrics (train/test) as JSON for dashboards and /api/metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_classification_metrics(
    *,
    model_name: str,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    target_names: list[str],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a dict suitable for JSON: accuracies, macro/weighted P/R/F1, report, confusion matrix."""
    target_names = [str(x) for x in target_names]
    acc_train = float(accuracy_score(y_train, y_train_pred))
    acc_test = float(accuracy_score(y_test, y_test_pred))

    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_test,
        y_test_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_test_pred, labels=list(range(len(target_names))))

    payload: dict[str, Any] = {
        "model_name": model_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f_m),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "classification_report_test": report,
        "confusion_matrix_test": cm.tolist(),
        "confusion_matrix_labels": target_names,
    }
    if extras:
        payload["extras"] = extras
    return _to_json_safe(payload)


def save_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
