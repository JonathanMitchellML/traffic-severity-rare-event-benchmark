"""Train baseline models for the traffic severity rare-event benchmark."""

from __future__ import annotations

import argparse
import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, make_modeling_table
from src.load_data import load_collision_data
from src.split import chronological_split, print_split_summary, split_summary
from src.utils import config_path, ensure_directory, load_config


def build_model_pipeline(X_train, model_config: dict) -> Pipeline:
    """Build the required class-weighted logistic regression baseline."""
    model_type = model_config.get("type", "logistic_regression")
    if model_type != "logistic_regression":
        raise ValueError(f"Unsupported model type for initial chassis: {model_type}")

    classifier = LogisticRegression(
        class_weight=model_config.get("class_weight", "balanced"),
        max_iter=int(model_config.get("max_iter", 500)),
        solver=model_config.get("solver", "saga"),
        random_state=int(model_config.get("random_seed", 42)),
    )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", classifier),
        ]
    )


def train_from_config(config: dict) -> dict:
    df = load_collision_data(config["dataset_path"])
    splits = chronological_split(df, config.get("split"))
    print_split_summary(splits)

    feature_config = config.get("features", {})
    include_factors = bool(feature_config.get("include_contributing_factors", False))
    include_street_names = bool(feature_config.get("include_street_names", False))
    X_train, y_train = make_modeling_table(
        splits["train"],
        include_contributing_factors=include_factors,
        include_street_names=include_street_names,
    )

    pipeline = build_model_pipeline(X_train, config.get("model", {}))
    pipeline.fit(X_train, y_train)

    model_path = config_path(
        config,
        "output",
        "model_path",
        default="artifacts/baseline_logistic_regression.joblib",
    )
    ensure_directory(model_path.parent)
    bundle = {
        "model": pipeline,
        "feature_columns": list(X_train.columns),
        "model_type": config.get("model", {}).get("type", "logistic_regression"),
        "include_contributing_factors": include_factors,
        "include_street_names": include_street_names,
        "target_definition": config.get("target", {}),
        "split_summary": split_summary(splits),
    }
    joblib.dump(bundle, model_path)

    metadata_path = model_path.with_suffix(".metadata.json")
    metadata = {k: v for k, v in bundle.items() if k != "model"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model artifact: {model_path}")
    print(f"Saved model metadata: {metadata_path}")
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline model.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_from_config(config)


if __name__ == "__main__":
    main()
