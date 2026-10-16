# src — data pipeline, baseline, and model tournament

Python scripts that take the raw dataset to the trained model that `api/` serves.

## Files

| File                                 | Purpose                                                                 |
|--------------------------------------|-------------------------------------------------------------------------|
| `clean.py`                           | Applies 6 corrections to the raw CSV; writes the cleaned CSV            |
| `baseline_random_classifier.py`      | Stratified-random `DummyClassifier` baseline (floor to beat)            |
| `model_tournament.py`                | 6-round bracket: baseline → LR → RF → HGB → MLP → XGB → RF-tuned        |
| `transaction-dataset.csv`            | Raw dataset (input to `clean.py`)                                       |
| `transaction-dataset-cleaned.csv`    | Cleaned dataset (output of `clean.py`, input to everything downstream)  |

## Run order

```bash
python clean.py                         # writes transaction-dataset-cleaned.csv
python baseline_random_classifier.py    # prints baseline macro-F1 (~8%)
python model_tournament.py              # runs the full bracket, prints winner
```

All scripts expect to be run **from this `src/` directory** — they read the CSVs
by filename.

## Dependencies

```bash
pip install pandas scikit-learn xgboost
```

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
