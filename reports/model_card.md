# Model Card

## Intended Use

This baseline is a reproducible benchmark for evaluating severe crash outcome prediction on public NYC crash records.

## Not Intended Use

This model is not intended for operational dispatch, enforcement, insurance decisions, or individual-level risk assessment.

## Dataset

NYC Motor Vehicle Collisions crash records, expected locally at `datasets/h9gi-nx95_2021_2024.csv`.

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
