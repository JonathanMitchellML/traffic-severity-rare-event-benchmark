import pandas as pd

from src.features import create_serious_event_target


def test_fatal_crash_becomes_target_one():
    df = pd.DataFrame(
        {
            "number_of_persons_killed": [1],
            "number_of_persons_injured": [0],
        }
    )

    target = create_serious_event_target(df)

    assert target.tolist() == [1]


def test_two_or_more_injuries_becomes_target_one():
    df = pd.DataFrame(
        {
            "number_of_persons_killed": [0, 0],
            "number_of_persons_injured": [2, 3],
        }
    )

    target = create_serious_event_target(df)

    assert target.tolist() == [1, 1]


def test_zero_killed_and_zero_or_one_injury_becomes_target_zero():
    df = pd.DataFrame(
        {
            "number_of_persons_killed": [0, 0],
            "number_of_persons_injured": [0, 1],
        }
    )

    target = create_serious_event_target(df)

    assert target.tolist() == [0, 0]


def test_missing_injury_and_killed_values_do_not_crash_target_creation():
    df = pd.DataFrame(
        {
            "number_of_persons_killed": [None, None, 1],
            "number_of_persons_injured": [None, 2, None],
        }
    )

    target = create_serious_event_target(df)

    assert target.tolist() == [0, 1, 1]

