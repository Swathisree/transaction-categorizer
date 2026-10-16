# src — data pipeline, baseline, and model tournament

Python scripts that take the raw dataset to the trained model that `api/` serves.
All inputs are passed via command-line arguments; the defaults point at
`../data/` so you can run each script from any working directory.

## Files

| File                                 | Purpose                                                       |
|--------------------------------------|---------------------------------------------------------------|
| `clean.py`                           | Applies 6 corrections to the raw CSV; writes the cleaned CSV  |
| `baseline_random_classifier.py`      | Stratified-random `DummyClassifier` baseline (floor to beat)  |
| `model_tournament.py`                | 6-round bracket: baseline → LR → RF → HGB → MLP → XGB         |

The CSVs themselves live in [`../data/`](../data/README.md) and are
gitignored (transaction data is private).

## Dependencies

```bash
pip install pandas scikit-learn xgboost
```

## Run order

Every script runs with sensible defaults — just invoke it. Pass `--input`
(and `--output` on `clean.py`) to override.

```bash
# 1. clean the raw dataset  →  ../data/transaction-dataset-cleaned.csv
python clean.py

# 2. baseline macro-F1 (~8%)
python baseline_random_classifier.py

# 3. full tournament
python model_tournament.py
```

Override the input path:

```bash
python clean.py --input /some/other/raw.csv --output /tmp/clean.csv
python baseline_random_classifier.py --input /tmp/clean.csv
python model_tournament.py --input /tmp/clean.csv
```

All three accept `--help` for the full argument list.

## What each round compared

| Round | Challenger                   | Champion coming in         | Winner |
|-------|------------------------------|----------------------------|--------|
| 1     | Logistic Regression          | Stratified random baseline | LR     |
| 2     | Random Forest                | Logistic Regression        | RF     |
| 3     | HistGradientBoosting         | Random Forest              | RF     |
| 4     | MLP (deep learning)          | Random Forest              | RF     |
| 5     | XGBoost                      | Random Forest              | RF     |
| 6     | Random Forest (grid-tuned)   | Random Forest (default)    | RF     |

Full write-ups for each round live in [`../docs/`](../docs/README.md).
