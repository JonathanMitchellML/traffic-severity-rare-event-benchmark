import pandas as pd
import pytest

from src.split import chronological_split


def _split_test_frame():
    return pd.DataFrame(
        {
            "crash_date": [
                "2024-06-01",
                "2021-01-01",
                "2023-07-01",
                "2022-12-31",
                "2024-01-01",
            ],
            "crash_time": ["10:00", "09:00", "12:00", "23:59", "00:01"],
            "number_of_persons_injured": [0, 0, 2, 1, 0],
            "number_of_persons_killed": [0, 0, 0, 0, 1],
            "collision_id": [100, 101, 102, 103, 104],
        },
        index=[10, 11, 12, 13, 14],
    )


def test_chronological_split_preserves_expected_time_order():
    splits = chronological_split(_split_test_frame())

    train_dates = pd.to_datetime(splits["train"]["crash_date"])
    validation_dates = pd.to_datetime(splits["validation"]["crash_date"])
    test_dates = pd.to_datetime(splits["test"]["crash_date"])

    assert train_dates.max() < validation_dates.min()
    assert validation_dates.max() < test_dates.min()
    assert train_dates.is_monotonic_increasing
    assert validation_dates.is_monotonic_increasing
    assert test_dates.is_monotonic_increasing


def test_chronological_split_has_no_overlapping_rows():
    splits = chronological_split(_split_test_frame())

    train_index = set(splits["train"].index)
    validation_index = set(splits["validation"].index)
    test_index = set(splits["test"].index)

    assert train_index.isdisjoint(validation_index)
    assert train_index.isdisjoint(test_index)
    assert validation_index.isdisjoint(test_index)


def test_chronological_split_raises_on_unparseable_crash_date():
    df = _split_test_frame()
    df.loc[11, "crash_date"] = "not-a-date"

    with pytest.raises(ValueError, match="crash_date"):
        chronological_split(df)

