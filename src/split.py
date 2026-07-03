"""Chronological split utilities for leakage-aware evaluation."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src.features import create_serious_event_target
from src.load_data import load_collision_data
from src.utils import load_config


DEFAULT_SPLIT_CONFIG = {
    "train_start": "2021-01-01",
    "train_end": "2022-12-31",
    "validation_start": "2023-01-01",
    "validation_end": "2023-12-31",
    "test_start": "2024-01-01",
    "test_end": "2024-12-31",
}


def _date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def chronological_split(
    df: pd.DataFrame,
    split_config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split rows into train/validation/test periods using crash_date.

    The split is explicitly chronological to avoid training on future crashes and
    evaluating on earlier records. Rows outside the configured windows are not
    included in any split.
    """
    config = {**DEFAULT_SPLIT_CONFIG, **(split_config or {})}
    if "crash_date" not in df.columns:
        raise ValueError("crash_date is required for chronological splitting")

    working = df.copy()
    working["_original_order"] = np.arange(len(working))
    working["_split_crash_date"] = pd.to_datetime(working["crash_date"], errors="coerce")
    if working["_split_crash_date"].isna().any():
        bad_count = int(working["_split_crash_date"].isna().sum())
        raise ValueError(f"crash_date has {bad_count:,} missing or unparseable values")

    sort_columns = ["_split_crash_date", "_original_order"]
    if "crash_datetime" in working.columns:
        working["_split_crash_datetime"] = pd.to_datetime(working["crash_datetime"], errors="coerce")
        if working["_split_crash_datetime"].notna().any():
            sort_columns = ["_split_crash_datetime", "_split_crash_date", "_original_order"]

    working = working.sort_values(sort_columns)
    crash_day = working["_split_crash_date"].dt.normalize()

    masks = {
        "train": crash_day.between(_date(config["train_start"]), _date(config["train_end"]), inclusive="both"),
        "validation": crash_day.between(
            _date(config["validation_start"]), _date(config["validation_end"]), inclusive="both"
        ),
        "test": crash_day.between(_date(config["test_start"]), _date(config["test_end"]), inclusive="both"),
    }

    helper_columns = [col for col in working.columns if col.startswith("_split_") or col == "_original_order"]
    return {name: working.loc[mask].drop(columns=helper_columns).copy() for name, mask in masks.items()}


def split_summary(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, float | int | str | None]]:
    """Summarize split sizes, date ranges, and target prevalence."""
    summary: dict[str, dict[str, float | int | str | None]] = {}
    for name, frame in splits.items():
        target = create_serious_event_target(frame) if len(frame) else pd.Series(dtype=int)
        dates = pd.to_datetime(frame["crash_date"], errors="coerce") if len(frame) else pd.Series(dtype="datetime64[ns]")
        summary[name] = {
            "rows": int(len(frame)),
            "target_rate": float(target.mean()) if len(target) else None,
            "target_count": int(target.sum()) if len(target) else 0,
            "min_date": str(dates.min().date()) if len(dates) else None,
            "max_date": str(dates.max().date()) if len(dates) else None,
        }
    return summary


def print_split_summary(splits: dict[str, pd.DataFrame]) -> None:
    for name, info in split_summary(splits).items():
        rate = "n/a" if info["target_rate"] is None else f"{info['target_rate']:.4%}"
        print(
            f"{name}: rows={info['rows']:,}, "
            f"target_rate={rate}, "
            f"date_range={info['min_date']} to {info['max_date']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create chronological train/validation/test splits.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    df = load_collision_data(config["dataset_path"])
    splits = chronological_split(df, config.get("split"))
    print_split_summary(splits)


if __name__ == "__main__":
    main()

