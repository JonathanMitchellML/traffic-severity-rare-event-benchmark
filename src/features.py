"""Leakage-aware target construction and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import normalize_column_names


TARGET_NAME = "serious_event"
TARGET_SOURCE_COLUMNS = [
    "number_of_persons_killed",
    "number_of_persons_injured",
]

LEAKAGE_COLUMNS = {
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_pedestrians_injured",
    "number_of_pedestrians_killed",
    "number_of_cyclist_injured",
    "number_of_cyclist_killed",
    "number_of_motorist_injured",
    "number_of_motorist_killed",
}

IDENTIFIER_COLUMNS = {"collision_id"}
RAW_LOCATION_COLUMNS = {"location"}

CONTRIBUTING_FACTOR_COLUMNS = [
    "contributing_factor_vehicle_1",
    "contributing_factor_vehicle_2",
    "contributing_factor_vehicle_3",
    "contributing_factor_vehicle_4",
    "contributing_factor_vehicle_5",
]

TEMPORAL_FEATURES = [
    "crash_hour",
    "crash_day_of_week",
    "crash_month",
    "is_weekend",
]

LOCATION_FEATURES = [
    "borough",
    "zip_code",
    "latitude",
    "longitude",
    "latitude_missing",
    "longitude_missing",
]

NUMERIC_FEATURES = [
    "crash_hour",
    "crash_day_of_week",
    "crash_month",
    "is_weekend",
    "latitude",
    "longitude",
    "latitude_missing",
    "longitude_missing",
]

STREET_FEATURES = [
    "on_street_name",
    "cross_street_name",
    "off_street_name",
]

VEHICLE_TYPE_FEATURES = [
    "vehicle_type_code_1",
    "vehicle_type_code_2",
    "vehicle_type_code_3",
    "vehicle_type_code_4",
    "vehicle_type_code_5",
]

BASE_FEATURE_CANDIDATES = (
    TEMPORAL_FEATURES + LOCATION_FEATURES + VEHICLE_TYPE_FEATURES
)


def create_serious_event_target(df: pd.DataFrame) -> pd.Series:
    """Construct the severe crash outcome indicator.

    Injury and fatality counts are allowed here only for target construction and
    diagnostics. Missing counts are treated as zero so target creation is stable
    on sparse or synthetic test data.
    """
    normalized = normalize_column_names(df)
    missing = [col for col in TARGET_SOURCE_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing target source columns: {missing}")

    killed = pd.to_numeric(normalized["number_of_persons_killed"], errors="coerce").fillna(0)
    injured = pd.to_numeric(normalized["number_of_persons_injured"], errors="coerce").fillna(0)
    target = ((killed > 0) | (injured >= 2)).astype(int)
    target.name = TARGET_NAME
    return target


def _derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    crash_date = pd.to_datetime(out["crash_date"], errors="coerce")

    if "crash_time" in out.columns:
        time_text = out["crash_time"].astype("string").str.strip()
        parsed_time = pd.to_datetime(time_text, format="%H:%M", errors="coerce")
        out["crash_hour"] = parsed_time.dt.hour
    elif "crash_hour" in out.columns:
        out["crash_hour"] = pd.to_numeric(out["crash_hour"], errors="coerce")
    else:
        out["crash_hour"] = np.nan

    out["crash_day_of_week"] = crash_date.dt.dayofweek
    out["crash_month"] = crash_date.dt.month
    out["is_weekend"] = np.where(
        crash_date.notna(),
        crash_date.dt.dayofweek.isin([5, 6]).astype(int),
        np.nan,
    )
    return out


def _clean_location_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["latitude", "longitude"]:
        if column not in out.columns:
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if "latitude" in out.columns:
        valid_latitude = out["latitude"].between(40.3, 41.1)
        out.loc[~valid_latitude, "latitude"] = np.nan
        out["latitude_missing"] = out["latitude"].isna().astype(int)

    if "longitude" in out.columns:
        valid_longitude = out["longitude"].between(-74.5, -73.2)
        out.loc[~valid_longitude, "longitude"] = np.nan
        out["longitude_missing"] = out["longitude"].isna().astype(int)

    return out


def _clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    categorical_columns = [
        col
        for col in out.columns
        if col not in {"latitude", "longitude", *TEMPORAL_FEATURES, "latitude_missing", "longitude_missing"}
    ]
    for column in categorical_columns:
        out[column] = out[column].astype("string").str.strip()
        out[column] = out[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def sanitize_for_sklearn(X: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable values/dtypes to sklearn-safe missing values.

    scikit-learn imputers handle ``np.nan`` reliably, while pandas ``pd.NA`` can
    trigger ambiguous boolean checks in object/string columns. This function is
    intentionally the final feature-table step before a dataframe enters a
    scikit-learn pipeline.
    """
    out = X.copy()
    for column in out.columns:
        if column in NUMERIC_FEATURES or pd.api.types.is_numeric_dtype(out[column]):
            out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
        else:
            series = out[column].astype("object")
            out[column] = series.where(series.notna(), np.nan)
    return out


def make_modeling_table(
    df: pd.DataFrame,
    *,
    include_contributing_factors: bool = False,
    include_street_names: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target y using conservative baseline features.

    The returned feature matrix deliberately omits all injury/fatality columns,
    collision identifiers, street-name fields, and contributing factor fields by
    default. Street names are high-cardinality context fields; contributing
    factors may encode officer-coded or post-report information.
    """
    working = normalize_column_names(df)
    y = create_serious_event_target(working)

    if "crash_date" not in working.columns:
        raise ValueError("crash_date is required to derive temporal features")

    working = _derive_temporal_features(working)
    working = _clean_location_features(working)

    candidate_features = list(BASE_FEATURE_CANDIDATES)
    if include_street_names:
        candidate_features.extend(STREET_FEATURES)
    if include_contributing_factors:
        candidate_features.extend(CONTRIBUTING_FACTOR_COLUMNS)

    available_features = [col for col in candidate_features if col in working.columns]
    X = working[available_features].copy()
    X = _clean_categorical_columns(X)

    for column in X.columns:
        if column in {"latitude", "longitude", *TEMPORAL_FEATURES, "latitude_missing", "longitude_missing"}:
            X[column] = pd.to_numeric(X[column], errors="coerce")

    blocked = (
        LEAKAGE_COLUMNS
        | IDENTIFIER_COLUMNS
        | RAW_LOCATION_COLUMNS
        | (set() if include_street_names else set(STREET_FEATURES))
        | (set() if include_contributing_factors else set(CONTRIBUTING_FACTOR_COLUMNS))
    )
    leaked = sorted(set(X.columns) & blocked)
    if leaked:
        raise ValueError(f"Leakage or blocked columns included in features: {leaked}")

    return sanitize_for_sklearn(X), y


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - compatibility for older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build train-fit-only preprocessing for numeric and categorical features."""
    numeric_features = [
        col
        for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])
    ]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No usable features were created for modeling")

    return ColumnTransformer(transformers=transformers, remainder="drop")
