# Lydia — transaction labeling technical test

Multi-class classification of Lydia card transactions into **minor_category**
(12 classes) and **major_category** (9 classes). The repo contains the full
pipeline — EDA, cleaning, baseline, a 6-round model tournament, and a FastAPI
server hosting the winning model.

```
.
├── data/    Transaction CSVs — raw + cleaned (gitignored, data is private)
├── src/     Python scripts (clean, baseline, tournament)
├── docs/    HTML reports, EDA, model-round write-ups, the test brief
└── api/     FastAPI prediction service for the trained model
```

Each sub-folder has its own README with internal details. Every training
script accepts `--input` / `--output` on the command line; the defaults
point at `data/` so no arguments are needed for the standard run.

---

## 1. Approach and key design choices

### Understanding of the problem
Each row is a card transaction with ~19 fields. The target column is
`minor_category` (12 values). `major_category` (9 values) is a **deterministic
function** of `minor_category` — every unique minor value maps to exactly one
major value — so only the minor task is actually learned. The major-category
prediction is produced by a lookup table built from the training data and
applied to the model's minor prediction. This avoids learning the same
information twice and keeps the two outputs mutually consistent by
construction.

### Key assumptions baked into the pipeline
- **`fees` is in cents in the raw dataset** (max value ~4000) while
  `amount_value` is in euros. The cleaning step rescales fees to euros so the
  two columns are directly comparable.
- **Refunds are a category, not a null**: 123 rows have
  `status = refunded` and no `major_category`. Those rows are given
  `major_category = "refund"` (adding one class to the major vocabulary) rather
  than dropped.
- **Missing `merchant_name` rows are ATM withdrawals**: all 113 nulls carry
  `title = "RETRAIT DAB"`. They are backfilled from `title`, which equals
  `merchant_name` 100% of the time where both are present.
- **Extreme outliers are data entry errors**, not real signal: 21 rows with
  `|amount_value| > €100,000` on fast-food / grocery merchants are dropped.
- **120 `ref_id` pairs are the same transaction ingested twice**; deduplication
  keeps the first row per `ref_id` (removes 118 rows after the outlier filter
  already consumed two of them).
- `card_name` (30 near-duplicate values) and `digitalization_type` (14 values,
  merchant leakage into the field) are left as-is — they still carry signal
  as categoricals, and normalising them is a separable task.

### Features and preprocessing
From the 19 raw columns the model consumes 9, grouped by how they are encoded:

| Group | Columns | Encoding |
|---|---|---|
| Numeric | `amount_value`, `fees`, `hour`, `dow` | `StandardScaler` |
| Low-cardinality categorical | `card_name`, `digitalization_type`, `merchant_pan_entry_method`, `status`, `pfm_recurring_frequency` | `OneHotEncoder` |
| High-cardinality categorical | `merchant_name` (959 unique) | multiclass `TargetEncoder` |

`created_at` is parsed into `hour` and `day-of-week` only — the 90-day window
is too short to learn calendar seasonality. `pfm_recurring_frequency` nulls
are coerced to the string `"none"` so they survive the one-hot encoder.

### Model selection: 6-round tournament
Rather than guess which model fits the dataset, each candidate competes
head-to-head on the same stratified 80/20 split (`random_state=42`). The
winner (by **minor macro F1**) stays as the base for the next round. All six
rounds, including per-class behaviour and confusion matrices, are in
`docs/` — full reproduction code in [src/model_tournament.py](src/model_tournament.py).

| Round | Challenger | Base | Winner |
|---|---|---|---|
| 1 | Logistic Regression | Stratified random baseline | Logistic Regression |
| 2 | Random Forest | Logistic Regression | **Random Forest** |
| 3 | HistGradientBoosting | Random Forest | Random Forest |
| 4 | MLP (deep learning) | Random Forest | Random Forest |
| 5 | XGBoost | Random Forest | Random Forest |
| 6 | Random Forest (grid-tuned) | Random Forest (default) | Random Forest (default) |

**Champion:** `RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
random_state=42, n_jobs=-1)`.

---

## 2. Result analysis

Headline metrics on the held-out test set (1,596 rows, stratified 20% split):

| Model | Minor macro F1 | Minor macro P | Minor macro R | Major macro F1 |
|---|---|---|---|---|
| Stratified random baseline | 8.18% | 8.17% | 8.20% | 10.64% |
| Logistic Regression | 87.73% | 94.14% | 84.51% | 88.89% |
| HistGradientBoosting | 87.28% | 91.31% | 84.90% | 88.61% |
| MLP (deep learning) | 85.45% | 87.01% | 84.09% | 86.02% |
| XGBoost | 87.73% | 91.04% | 85.62% | 89.14% |
| Random Forest (grid-tuned) | 87.72% | 91.85% | 85.32% | 89.10% |
| **Random Forest (default)** | **88.23%** | **93.93%** | **85.38%** | **89.47%** |

What the numbers say:
- The baseline floor is `mean(class priors) ≈ 8%` — every learned model
  clears it by 10× or more. The features carry real signal.
- **Precision > Recall** by 6–10 pp across the board. The models are
  conservative on long-tail classes (gambling, finance_fee, financial_services)
  — they predict them only when confident, so they miss some positives but
  rarely false-alarm.
- Gradient boosting (HGB, XGBoost) and the grid-tuned RF are within 0.3 pp of
  the default RF — the dataset is near its ceiling on this feature set. Any
  further gain is more likely to come from richer `merchant_name` features
  (character n-grams, embeddings) than from more hyperparameter tuning.
- The hyperparameter grid search in round 6 won on CV (88.14%) but lost on
  the held-out test (87.98% vs the default RF's 88.23%) — a textbook case of
  mild CV overfitting; the full round 6 report walks through the evidence.

Per-class F1, confusion matrices, and the full discussion for each round live
in [docs/](docs/README.md). The tournament overview is in
[docs/model_tournament_overview.html](docs/model_tournament_overview.html).

---

## 3. How to run training

Everything runs from Python 3.11+. Install the training dependencies:

```bash
pip install pandas scikit-learn xgboost
```

Place your copy of the raw CSV at `data/transaction-dataset.csv` (the
file is gitignored, so each contributor supplies their own).

**Full pipeline — from the raw CSV to the production model artifact:**

```bash
# 1. clean the raw dataset  →  data/transaction-dataset-cleaned.csv
python src/clean.py
# or if dataset is not inside data folder
python src/clean.py --input <path-to-the-data>

# 2. (optional) see where the baseline floor sits
python src/baseline_random_classifier.py

# 3. (optional) replay the full 6-round tournament
python src/model_tournament.py

# 4. train the champion RF on the full cleaned dataset and emit the
#    artifacts the API loads (api/model.joblib + api/model_meta.json)
python api/train_model.py
```

Every script accepts `--input /path/to/clean.csv` (and `clean.py` also
accepts `--output`) to point at a different CSV; the defaults resolve to
`data/` from the script's own location, so run them from anywhere.

The trained artifacts land in `api/`:

- `model.joblib` — serialized sklearn Pipeline (preprocessor + Random Forest).
- `model_meta.json` — minor↔major lookup table, class vocabularies, and the
  list of raw input columns the API expects.

`model.joblib` is already committed to the repo, so step 3 is only needed
if you want to retrain against a modified cleaned dataset.

---

## 4. How to run the prediction service

### Install

```bash
cd api
pip install -r requirements.txt
```

### Start the server

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t transaction-categorizer-api .
docker run -p 8000:8000 transaction-categorizer-api
```

The API logs the loaded model name and class counts on startup, then listens
on `http://127.0.0.1:8000`. Swagger UI is at `/docs`, ReDoc at `/redoc`.

### Endpoints

| Method | Path             | Purpose                                       |
|--------|------------------|-----------------------------------------------|
| GET    | `/`              | Landing page                                  |
| GET    | `/health`        | Liveness + model metadata                     |
| GET    | `/categories`    | Supported minor / major classes + lookup      |
| POST   | `/predict`       | Single-transaction prediction                 |
| POST   | `/predict/batch` | Batch prediction (list of transactions)       |

### Request format

A single-prediction request is a JSON object with these fields:

| Field | Type | Notes |
|---|---|---|
| `amount_value` | number (EUR) | Negative = debit, positive = refund |
| `fees` | number (EUR) | 0 for most rows |
| `created_at` | ISO 8601 timestamp | Parsed to `hour` + `day-of-week` server-side |
| `card_name` | string | e.g. `"Carte Lydia bleue"` |
| `digitalization_type` | string | e.g. `"apple_pay"`, `"none"` |
| `merchant_pan_entry_method` | string | e.g. `"CHIP_CONTACTLESS"`, `"CARD_ON_FILE"` |
| `status` | string | `"completed"` or `"refunded"` |
| `pfm_recurring_frequency` | string \| null | `"monthly"` or `null` |
| `merchant_name` | string | Main signal — target-encoded by the model |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount_value": -22.18,
    "fees": 0.0,
    "created_at": "2026-04-28T11:45:21Z",
    "card_name": "Carte sans contact basique",
    "digitalization_type": "apple_pay",
    "merchant_pan_entry_method": "CHIP_CONTACTLESS",
    "status": "completed",
    "pfm_recurring_frequency": null,
    "merchant_name": "MAC DONALD S"
  }'
```

### Response format

```json
{
  "predicted_minor_category": "restaurants",
  "predicted_major_category": "restaurants_bars_cafes",
  "confidence": 0.9250,
  "top3_minor": [
    {"minor_category": "restaurants",     "probability": 0.9250},
    {"minor_category": "supermarkets",    "probability": 0.0643},
    {"minor_category": "transport_means", "probability": 0.0051}
  ]
}
```

For batch prediction, POST a JSON array of the same objects to
`/predict/batch`; the response is a parallel array of the same prediction
objects.
