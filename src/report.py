"""Generate markdown reports and figures from evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.utils import config_path, ensure_directory, load_config


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.2f}%"


def _load_results(config: dict) -> dict[str, Any]:
    evaluation_path = config_path(config, "output", "evaluation_path", default="artifacts/evaluation_results.json")
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation results not found. Run evaluation first: {evaluation_path}")
    return json.loads(evaluation_path.read_text(encoding="utf-8"))


def _plot_pr_curve(predictions: pd.DataFrame, title: str, path: Path, prevalence: float | None) -> None:
    precision, recall, _ = precision_recall_curve(predictions["y_true"], predictions["score"])
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2)
    if prevalence is not None:
        plt.axhline(
            prevalence,
            linestyle="--",
            color="gray",
            linewidth=1.5,
            label=f"Target prevalence ({_pct(prevalence)})",
        )
        plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_confusion_matrix(matrix: dict[str, int], path: Path) -> None:
    values = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    plt.figure(figsize=(5, 4))
    plt.imshow(values, cmap="Blues")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            plt.text(col_idx, row_idx, f"{value:,}", ha="center", va="center", color="black")
    plt.title("Test Confusion Matrix")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_threshold_sweep(sweep: list[dict[str, float]], path: Path) -> None:
    frame = pd.DataFrame(sweep)
    plt.figure(figsize=(7, 5))
    plt.plot(frame["threshold"], frame["precision"], label="Precision")
    plt.plot(frame["threshold"], frame["recall"], label="Recall")
    plt.plot(frame["threshold"], frame["f1"], label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Validation Threshold Sweep")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _split_table(results: dict[str, Any]) -> str:
    rows = [
        "| Split | Rows | Date Window | Target Rate | Target Count |",
        "|---|---:|---|---:|---:|",
    ]
    for name in ["train", "validation", "test"]:
        info = results["splits"][name]
        rows.append(
            f"| {name} | {info['rows']:,} | {info['min_date']} to {info['max_date']} | "
            f"{_pct(info['target_rate'])} | {info['target_count']:,} |"
        )
    return "\n".join(rows)


def _metrics_table(results: dict[str, Any]) -> str:
    rows = [
        "| Split | PR-AUC / Avg Precision | ROC-AUC | Precision | Recall | F1 | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["validation", "test"]:
        metrics = results[name]
        rows.append(
            f"| {name} | {_fmt(metrics['average_precision'])} | {_fmt(metrics['roc_auc'])} | "
            f"{_fmt(metrics['precision'])} | {_fmt(metrics['recall'])} | {_fmt(metrics['f1'])} | "
            f"{_fmt(metrics['threshold'], 3)} |"
        )
    return "\n".join(rows)


def write_evaluation_summary(results: dict[str, Any], report_path: Path) -> None:
    split_info = results["splits"]
    dataset_window = f"{split_info['train']['min_date']} through {split_info['test']['max_date']}"
    target_definition = results["target"].get(
        "definition",
        "number_of_persons_killed > 0 or number_of_persons_injured >= 2",
    )

    street_policy = results.get("feature_policy", {})
    if street_policy.get("street_names_included", False):
        feature_runtime_policy = (
            "Street-name fields are enabled for this run. The baseline feature set also uses crash time, "
            "calendar features, borough/ZIP/location coordinates when valid, and vehicle type codes."
        )
    else:
        feature_runtime_policy = (
            "Street-name fields are excluded by default because they are high-cardinality fields for "
            "the initial baseline. The baseline feature set instead uses crash time, calendar features, "
            "borough/ZIP/location coordinates when valid, and vehicle type codes."
        )

    body = f"""# Evaluation Summary

## Dataset Window

{dataset_window}

## Target Definition

`serious_event = 1` when `{target_definition}`.

## Split Design

{_split_table(results)}

## Model

Baseline model: `{results['model_type']}` with train-fit-only preprocessing.

## Validation Threshold

Threshold selected on validation only: `{results['selected_validation_threshold']:.3f}`.

## Metrics

{_metrics_table(results)}

## Test Confusion Matrix

TN: {results['test']['confusion_matrix']['tn']:,}  
FP: {results['test']['confusion_matrix']['fp']:,}  
FN: {results['test']['confusion_matrix']['fn']:,}  
TP: {results['test']['confusion_matrix']['tp']:,}

## Leakage Exclusions

The baseline excludes injury/fatality count fields from model features after target construction. It also excludes collision identifiers and contributing factor fields by default because contributing factors may represent officer-coded or post-report information.

## Feature Runtime Policy

{feature_runtime_policy}
"""
    report_path.write_text(body, encoding="utf-8")


def write_model_card(results: dict[str, Any], report_path: Path) -> None:
    body = f"""# Model Card

## Intended Use

This baseline is a reproducible benchmark for evaluating severe crash outcome prediction on public NYC crash records.

## Not Intended Use

This model is not intended for operational dispatch, enforcement, insurance decisions, or individual-level risk assessment.

## Dataset

NYC Motor Vehicle Collisions crash records, expected locally at `{results['dataset_path']}`.

## Target

`serious_event` indicates at least one fatality or at least two reported injuries.

## Features

The initial feature set uses crash time, calendar features, borough/ZIP/location coordinates when valid, and vehicle type codes. Street-name fields are excluded by default because they are high-cardinality fields for an initial baseline.

## Leakage Controls

Injury/fatality counts are used only to construct and diagnose the target. Collision IDs, raw location strings, and contributing factor fields are excluded from the default baseline.

## Evaluation Design

Training uses 2021-2022 records, validation uses 2023 records, and testing uses 2024 records. The classification threshold is selected on validation F1 and applied once to test.

## Limitations

Police-reported crash data can contain missing, delayed, or revised fields. The baseline is intentionally simple and does not prove causal relationships or operational readiness.

## Ethical / Operational Caution

Rare-event traffic models can reinforce reporting biases and should not be used without domain review, monitoring, and clear accountability.
"""
    report_path.write_text(body, encoding="utf-8")


def generate_reports(config: dict) -> None:
    results = _load_results(config)
    reports_dir = ensure_directory(config.get("output", {}).get("reports_dir", "reports"))
    figures_dir = ensure_directory(config.get("output", {}).get("figures_dir", "reports/figures"))

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

    validation_predictions = pd.read_csv(validation_predictions_path)
    test_predictions = pd.read_csv(test_predictions_path)

    _plot_pr_curve(
        validation_predictions,
        "Validation Precision-Recall Curve",
        figures_dir / "validation_pr_curve.png",
        results["splits"]["validation"]["target_rate"],
    )
    _plot_pr_curve(
        test_predictions,
        "Test Precision-Recall Curve",
        figures_dir / "test_pr_curve.png",
        results["splits"]["test"]["target_rate"],
    )
    _plot_confusion_matrix(results["test"]["confusion_matrix"], figures_dir / "test_confusion_matrix.png")
    _plot_threshold_sweep(results["threshold_sweep"], figures_dir / "validation_threshold_sweep.png")

    write_evaluation_summary(results, reports_dir / "evaluation_summary.md")
    write_model_card(results, reports_dir / "model_card.md")

    print(f"Saved evaluation summary: {reports_dir / 'evaluation_summary.md'}")
    print(f"Saved model card: {reports_dir / 'model_card.md'}")
    print(f"Saved figures under: {figures_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark reports and figures.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    generate_reports(config)


if __name__ == "__main__":
    main()
