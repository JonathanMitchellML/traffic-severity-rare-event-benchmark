# Traffic Severity Rare-Event Benchmark

This project is a reproducible, leakage-aware machine learning benchmark for predicting severe traffic crash outcomes from public NYC Motor Vehicle Collisions crash records.

The goal is not leaderboard-style model performance. The goal is to show professional applied ML judgment: explicit target definition, chronological splitting, leakage controls, rare-event metrics, reproducible reporting, tests, and readable scripts.

## Why This Project Exists

Traffic crash severity prediction is a realistic rare-event modeling problem. Severe outcomes are uncommon, easy to overstate with the wrong metric, and vulnerable to leakage if injury, fatality, or post-report fields are used incorrectly. This repository demonstrates a conservative baseline workflow that a technical reviewer can inspect end to end.

## Dataset

Source: NYC Motor Vehicle Collisions - Crashes.

Expected local file:

```text
datasets/h9gi-nx95_2021_2024.csv
```

The raw CSV is not committed to this repository. `.gitignore` excludes `datasets/*.csv`.

## Target Definition

The binary target is `serious_event`:

```python
serious_event = (
    number_of_persons_killed > 0
    or number_of_persons_injured >= 2
)
```

This is framed as a severe crash outcome indicator. Injury and fatality count fields are used only for target construction and target diagnostics. They are not used as model features.

## Leakage-Aware Design Choices

The baseline feature set is intentionally conservative:

- Crash temporal features: hour, day of week, month, weekend indicator.
- Location/context features: borough, ZIP code, latitude/longitude when valid, and missing-location flags.
- Vehicle type code fields.

The baseline excludes:

- All injury and fatality count fields after target construction.
- `collision_id`.
- Raw `location` string, because latitude/longitude are handled directly.
- Street-name fields by default, because they are high-cardinality context fields for an initial baseline. They can be enabled with `features.include_street_names`.
- Contributing factor fields by default, because they may represent officer-coded or post-report information that is not available at prediction time.

Preprocessing is handled inside scikit-learn pipelines so imputers, encoders, and scalers are fit on training data only.

## Chronological Split

The default split is fixed for the 2021-2024 local dataset:

| Split | Dates |
|---|---|
| Train | 2021-01-01 through 2022-12-31 |
| Validation | 2023-01-01 through 2023-12-31 |
| Test | 2024-01-01 through 2024-12-31 |

The classification threshold is selected on validation F1 only and then applied once to the test split.

## Metrics

The evaluation emphasizes rare-event metrics:

- PR-AUC / average precision.
- Precision, recall, and F1 at the selected threshold.
- F1 threshold sweep on validation.
- Confusion matrix.
- Target prevalence by split.

ROC-AUC is reported when both classes are present, but it should not be the main decision metric for this rare-event problem.

## How To Run From PowerShell

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest
```

Inspect the dataset:

```powershell
python -m src.load_data --config configs/baseline.yaml
```

Train, evaluate, and generate reports:

```powershell
python -m src.train --config configs/baseline.yaml
python -m src.evaluate --config configs/baseline.yaml
python -m src.report --config configs/baseline.yaml
```

If `make` is available, the same flow is available as:

```powershell
make install
make test
make run
```

## Expected Outputs

Generated files are ignored by Git:

```text
artifacts/baseline_logistic_regression.joblib
artifacts/baseline_logistic_regression.metadata.json
artifacts/evaluation_results.json
artifacts/validation_predictions.csv
artifacts/test_predictions.csv
reports/evaluation_summary.md
reports/model_card.md
reports/figures/
```

## Docker

The Dockerfile installs dependencies and runs tests. It does not copy the raw dataset into the image.

```powershell
docker build -t traffic-severity-rare-event-benchmark .
docker run --rm traffic-severity-rare-event-benchmark
```

To run the full pipeline in Docker, mount a local dataset directory and override or match the configured dataset path.

## Limitations

- The first baseline is deliberately simple and not optimized for production use.
- Contributing factor fields are excluded by default; using them requires a manual review of prediction-time availability.
- Police-reported crash data can contain missing, delayed, revised, or biased records.
- The target is a practical severe-outcome proxy, not a complete public safety objective.
- This benchmark does not establish causality.

## Development Note

Development note: This project was developed with AI-assisted coding support. The problem framing, target definition, leakage policy, evaluation design, code review, and final validation are author-owned.
