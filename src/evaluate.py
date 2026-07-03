"""Evaluate trained models with validation-selected thresholds."""

from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import (
    CONTRIBUTING_FACTOR_COLUMNS,
    LEAKAGE_COLUMNS,
    STREET_FEATURES,
    make_modeling_table,
    sanitize_for_sklearn,
)
from src.load_data import load_collision_data
from src.split import chronological_split, split_summary
from src.utils import config_path, ensure_directory, load_config


def _has_both_classes(y_true: pd.Series | np.ndarray) -> bool:
    return len(np.unique(np.asarray(y_true))) == 2


def _score_model(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        return 1 / (1 + np.exp(-raw_scores))
    raise TypeError("Model must expose predict_proba or decision_function")


def threshold_sweep(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Evaluate precision/recall/F1 over candidate thresholds."""
    if thresholds is None:
        thresholds = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.array([0.5])]))

    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return rows


def select_best_f1_threshold(sweep: list[dict[str, float]]) -> dict[str, float]:
    """Select a threshold using validation F1 only."""
    return max(sweep, key=lambda row: (row["f1"], row["recall"], row["precision"]))


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "average_precision": float(average_precision_score(y_true, y_score)) if np.sum(y_true) > 0 else None,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if _has_both_classes(y_true) else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _write_prediction_file(path, y_true: pd.Series, y_score: np.ndarray) -> None:
    ensure_directory(path.parent)
    pd.DataFrame({"y_true": np.asarray(y_true).astype(int), "score": y_score}).to_csv(path, index=False)


def evaluate_from_config(config: dict) -> dict[str, Any]:
    model_path = config_path(
        config,
        "output",
        "model_path",
        default="artifacts/baseline_logistic_regression.joblib",
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found. Run training first: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    include_factors = bool(bundle.get("include_contributing_factors", False))
    include_street_names = bool(bundle.get("include_street_names", False))

    df = load_collision_data(config["dataset_path"])
    splits = chronological_split(df, config.get("split"))

    modeling = {
        name: make_modeling_table(
            frame,
            include_contributing_factors=include_factors,
            include_street_names=include_street_names,
        )
        for name, frame in splits.items()
    }
    X_validation, y_validation = modeling["validation"]
    X_test, y_test = modeling["test"]
    X_validation = sanitize_for_sklearn(X_validation.reindex(columns=feature_columns))
    X_test = sanitize_for_sklearn(X_test.reindex(columns=feature_columns))

    validation_scores = _score_model(model, X_validation)
    test_scores = _score_model(model, X_test)
    sweep = threshold_sweep(y_validation, validation_scores)
    selected = select_best_f1_threshold(sweep)
    selected_threshold = selected["threshold"]

    results = {
        "dataset_path": config["dataset_path"],
        "target": config.get("target", {}),
        "model_type": bundle.get("model_type", config.get("model", {}).get("type", "logistic_regression")),
        "selected_validation_threshold": selected_threshold,
        "selected_validation_threshold_metrics": selected,
        "splits": split_summary(splits),
        "validation": metrics_at_threshold(y_validation, validation_scores, selected_threshold),
        "test": metrics_at_threshold(y_test, test_scores, selected_threshold),
        "threshold_sweep": sweep,
        "leakage_exclusions": sorted(LEAKAGE_COLUMNS),
        "feature_policy": {
            "street_names_included": include_street_names,
            "street_name_columns": STREET_FEATURES,
            "street_name_note": (
                "Street-name fields are excluded by default because they are high-cardinality fields "
                "for the initial baseline."
            ),
        },
        "contributing_factor_policy": {
            "included": include_factors,
            "columns": CONTRIBUTING_FACTOR_COLUMNS,
            "default_note": "Excluded by default because they can represent officer-coded or post-report information.",
        },
    }

    evaluation_path = config_path(config, "output", "evaluation_path", default="artifacts/evaluation_results.json")
    ensure_directory(evaluation_path.parent)
    evaluation_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    validation_predictions_path = config_path(
        config,
        "output",
        "validation_predictions_path",
        default="artifacts/validation_predictions.csv",
    )
    test_predictions_path = config_path(
        config,
        "output",
        "test_predictions_path",
        default="artifacts/test_predictions.csv",
    )
    _write_prediction_file(validation_predictions_path, y_validation, validation_scores)
    _write_prediction_file(test_predictions_path, y_test, test_scores)

    print(f"Selected validation threshold: {selected_threshold:.3f}")
    print(f"Validation F1 at selected threshold: {results['validation']['f1']:.4f}")
    print(f"Test F1 at selected threshold: {results['test']['f1']:.4f}")
    print(f"Saved evaluation results: {evaluation_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline model.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    evaluate_from_config(config)


if __name__ == "__main__":
    main()
