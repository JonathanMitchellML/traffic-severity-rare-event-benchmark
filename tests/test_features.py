import pandas as pd

from src.features import (
    CONTRIBUTING_FACTOR_COLUMNS,
    LEAKAGE_COLUMNS,
    STREET_FEATURES,
    build_preprocessor,
    make_modeling_table,
)


def _tiny_crash_frame():
    return pd.DataFrame(
        {
            "crash_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "crash_time": ["08:15", "17:30", None],
            "borough": ["BROOKLYN", "QUEENS", None],
            "zip_code": ["11201", "11375", None],
            "latitude": [40.7, 0, None],
            "longitude": [-73.9, 0, None],
            "location": ["(40.7, -73.9)", "(0, 0)", None],
            "on_street_name": ["ATLANTIC AVE", "QUEENS BLVD", None],
            "cross_street_name": ["COURT ST", None, None],
            "off_street_name": [None, None, "PARKING LOT"],
            "number_of_persons_injured": [0, 2, None],
            "number_of_persons_killed": [0, 0, None],
            "number_of_pedestrians_injured": [0, 1, None],
            "number_of_pedestrians_killed": [0, 0, None],
            "collision_id": [1, 2, 3],
            "contributing_factor_vehicle_1": ["Driver Inattention/Distraction", None, None],
            "vehicle_type_code1": ["Sedan", "Taxi", None],
            "vehicle_type_code2": ["Bike", None, None],
        }
    )


def test_modeling_table_excludes_target_leakage_and_default_post_report_fields():
    X, y = make_modeling_table(_tiny_crash_frame())

    assert y.tolist() == [0, 1, 0]
    assert not (set(X.columns) & LEAKAGE_COLUMNS)
    assert "collision_id" not in X.columns
    assert "location" not in X.columns
    assert not (set(X.columns) & set(CONTRIBUTING_FACTOR_COLUMNS))
    assert not (set(X.columns) & set(STREET_FEATURES))


def test_modeling_table_creates_expected_safe_features():
    X, _ = make_modeling_table(_tiny_crash_frame())

    expected = {
        "crash_hour",
        "crash_day_of_week",
        "crash_month",
        "is_weekend",
        "borough",
        "zip_code",
        "latitude",
        "longitude",
        "latitude_missing",
        "longitude_missing",
        "vehicle_type_code_1",
        "vehicle_type_code_2",
    }
    assert expected.issubset(set(X.columns))
    assert X.loc[1, "latitude_missing"] == 1
    assert X.loc[1, "longitude_missing"] == 1


def test_street_name_fields_are_optional_high_cardinality_features():
    X, _ = make_modeling_table(_tiny_crash_frame(), include_street_names=True)

    assert set(STREET_FEATURES).issubset(set(X.columns))


def test_nullable_pandas_values_are_safe_for_sklearn_preprocessing():
    df = pd.DataFrame(
        {
            "crash_date": ["2021-01-01", "2021-01-02", pd.NA, "2021-01-04"],
            "crash_time": ["08:15", pd.NA, "20:30", "bad-time"],
            "borough": pd.Series(["BROOKLYN", pd.NA, "QUEENS", ""], dtype="string"),
            "zip_code": pd.Series(["11201", pd.NA, "11375", ""], dtype="string"),
            "latitude": pd.Series([40.7, pd.NA, 40.8, 0], dtype="Float64"),
            "longitude": pd.Series([-73.9, pd.NA, -73.8, 0], dtype="Float64"),
            "number_of_persons_injured": pd.Series([0, 2, pd.NA, 1], dtype="Int64"),
            "number_of_persons_killed": pd.Series([0, 0, 1, pd.NA], dtype="Int64"),
            "vehicle_type_code1": pd.Series(["Sedan", pd.NA, "Taxi", ""], dtype="string"),
            "vehicle_type_code2": pd.Series([pd.NA, "Bike", "", "Bus"], dtype="string"),
        }
    )
    X, _ = make_modeling_table(df)

    transformed = build_preprocessor(X).fit_transform(X)

    assert transformed.shape[0] == len(df)
