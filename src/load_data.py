"""Load and lightly validate the NYC crash dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import load_config, normalize_column_names, resolve_project_path


REQUIRED_COLUMNS = {
    "crash_date",
    "crash_time",
    "number_of_persons_injured",
    "number_of_persons_killed",
}


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if the minimum source columns are unavailable."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")


def parse_crash_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse crash date/time and derive a crash_datetime helper column.

    Parsing is safe and non-imputing: invalid dates become NaT, and invalid times
    leave crash_datetime at the date-only value. Split validation later raises if
    crash_date itself is missing or unparseable.
    """
    out = df.copy()
    out["crash_date"] = pd.to_datetime(out["crash_date"], errors="coerce")

    time_text = out["crash_time"].astype("string").str.strip()
    parsed_time = pd.to_datetime(time_text, format="%H:%M", errors="coerce")
    out["crash_hour"] = parsed_time.dt.hour
    out["crash_minute"] = parsed_time.dt.minute

    out["crash_datetime"] = out["crash_date"].dt.normalize()
    has_date_and_time = out["crash_date"].notna() & parsed_time.notna()
    out.loc[has_date_and_time, "crash_datetime"] = (
        out.loc[has_date_and_time, "crash_date"].dt.normalize()
        + pd.to_timedelta(parsed_time.loc[has_date_and_time].dt.hour, unit="h")
        + pd.to_timedelta(parsed_time.loc[has_date_and_time].dt.minute, unit="m")
    )
    return out


def load_collision_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the local crash CSV and apply safe parsing/column normalization."""
    resolved = resolve_project_path(csv_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    df = pd.read_csv(resolved, low_memory=False)
    df = normalize_column_names(df)
    validate_required_columns(df)
    return parse_crash_datetime(df)


def print_diagnostics(df: pd.DataFrame) -> None:
    """Print basic diagnostics for local sanity checks."""
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Crash date min: {df['crash_date'].min()}")
    print(f"Crash date max: {df['crash_date'].max()}")

    try:
        from src.features import create_serious_event_target

        target = create_serious_event_target(df)
        print(f"Serious event rate: {target.mean():.4%}")
        print(f"Serious event count: {int(target.sum()):,}")
    except Exception as exc:  # pragma: no cover - diagnostic path only
        print(f"Target diagnostics unavailable: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and inspect the crash dataset.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data", default=None, help="Optional dataset path override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_path = args.data or config.get("dataset_path")
    if not dataset_path:
        raise ValueError("Dataset path must be provided via --data or config.dataset_path")

    df = load_collision_data(dataset_path)
    print_diagnostics(df)


if __name__ == "__main__":
    main()

