# Evaluation Summary

## Dataset Window

2021-01-01 through 2024-12-31

## Target Definition

`serious_event = 1` when `number_of_persons_killed > 0 or number_of_persons_injured >= 2`.

## Split Design

| Split | Rows | Date Window | Target Rate | Target Count |
|---|---:|---|---:|---:|
| train | 214,445 | 2021-01-01 to 2022-12-31 | 8.18% | 17,534 |
| validation | 96,607 | 2023-01-01 to 2023-12-31 | 9.46% | 9,142 |
| test | 91,316 | 2024-01-01 to 2024-12-31 | 10.04% | 9,167 |

## Model

Baseline model: `logistic_regression` with train-fit-only preprocessing.

## Validation Threshold

Threshold selected on validation only: `0.580`.

## Metrics

| Split | PR-AUC / Avg Precision | ROC-AUC | Precision | Recall | F1 | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.1863 | 0.6967 | 0.1756 | 0.4957 | 0.2594 | 0.580 |
| test | 0.1950 | 0.6961 | 0.1846 | 0.4660 | 0.2644 | 0.580 |

## Test Confusion Matrix

TN: 63,279  
FP: 18,870  
FN: 4,895  
TP: 4,272

## Leakage Exclusions

The baseline excludes injury/fatality count fields from model features after target construction. It also excludes collision identifiers and contributing factor fields by default because contributing factors may represent officer-coded or post-report information.

## Feature Runtime Policy

Street-name fields are excluded by default because they are high-cardinality fields for the initial baseline. The baseline feature set instead uses crash time, calendar features, borough/ZIP/location coordinates when valid, and vehicle type codes.
