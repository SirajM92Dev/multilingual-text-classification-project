from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[int] | None = None,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def metrics_by_language(
    frame: pd.DataFrame,
    true_column: str = "label_id",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    rows = []
    for language, group in frame.groupby("language"):
        result = classification_metrics(group[true_column], group[prediction_column])
        rows.append(
            {
                "language": language,
                "samples": len(group),
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "f1_weighted": result["f1_weighted"],
                "precision_macro": result["precision_macro"],
                "recall_macro": result["recall_macro"],
            }
        )
    return pd.DataFrame(rows).sort_values("language").reset_index(drop=True)
