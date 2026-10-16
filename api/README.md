# Transaction categorizer API

[![ci](https://img.shields.io/badge/ci-github--actions-blue)](./.github/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11-blue)](#)
[![license](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

A FastAPI server that exposes a Random Forest (the champion of a 6-round model tournament) for categorizing Lydia card transactions. Each request returns both a **minor_category** (12 classes) and a **major_category** (9 classes, derived from the minor prediction via a deterministic lookup).

- Minor macro F1 on held-out test set: **88.23%**
- Major macro F1 on held-out test set: **89.50%**
- Model trained on the full cleaned dataset (7,977 transactions)

---

## Quick start — local

```bash
pip install -r requirements.txt
python -m uvicorn api:app --reload --port 8000
```

Then:

- http://127.0.0.1:8000/ — landing page
- http://127.0.0.1:8000/docs — Swagger UI (interactive)
- http://127.0.0.1:8000/redoc — ReDoc

In another terminal, hit it with the example client:

```bash
python client_example.py
```

## Quick start — Docker

```bash
docker build -t transaction-categorizer-api .
docker run -p 8000:8000 transaction-categorizer-api
```

Or with Compose:

```bash
docker compose up --build
```

## Endpoints

| Method | Path             | Purpose                                       |
|--------|------------------|-----------------------------------------------|
| GET    | `/`              | HTML landing page                             |
| GET    | `/health`        | Liveness check and model metadata             |
| GET    | `/categories`    | Supported minor + major categories and lookup |
| POST   | `/predict`       | Single-transaction prediction                 |
| POST   | `/predict/batch` | Batch prediction (list of transactions)       |

### Input schema

| field                       | type               | notes                                        |
|-----------------------------|--------------------|----------------------------------------------|
| `amount_value`              | number (EUR)       | negative = debit, positive = refund          |
| `fees`                      | number (EUR)       | 0 for most rows                              |
| `created_at`                | ISO 8601 timestamp | used to derive hour + day-of-week internally |
| `card_name`                 | string             | e.g. `"Carte Lydia bleue"`                   |
| `digitalization_type`       | string             | e.g. `"apple_pay"`, `"none"`                 |
| `merchant_pan_entry_method` | string             | e.g. `"CHIP_CONTACTLESS"`, `"CARD_ON_FILE"`  |
| `status`                    | string             | `"completed"` or `"refunded"`                |
| `pfm_recurring_frequency`   | string \| null     | `"monthly"` or `null`                        |
| `merchant_name`             | string             | main signal — target-encoded by the model    |

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

## Repo layout

```
.
├── .github/workflows/ci.yml   GitHub Actions: smoke-test + docker-build
├── .gitignore                 CSVs are gitignored (transaction data is private)
├── Dockerfile                 python:3.11-slim · uvicorn on port 8000
├── docker-compose.yml         single-service compose definition
├── LICENSE                    MIT
├── README.md                  this file
├── api.py                     FastAPI server
├── client_example.py          urllib-only client that hits every endpoint
├── model.joblib               serialized sklearn Pipeline (preprocessor + RF)
├── model_meta.json            minor → major lookup + supported classes
├── requirements.txt           runtime deps
└── train_model.py             regenerate model.joblib from the cleaned CSV
```

> **Note.** The cleaned CSV (`transaction-dataset-cleaned.csv`) is **not** committed — transaction data is private. Point `train_model.py` at your own copy to regenerate the model.

## Model

- `RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)`
- Preprocessor (`ColumnTransformer`):
  - `StandardScaler` on numerics (`amount_value`, `fees`, `hour`, `dow`)
  - `OneHotEncoder` on low-cardinality categoricals (`card_name`, `digitalization_type`, `merchant_pan_entry_method`, `status`, `pfm_recurring_frequency`)
  - multiclass `TargetEncoder` on `merchant_name`

Tournament leaderboard it topped (minor macro F1 on an 80/20 stratified test split):

| Rank | Model                        | minor macro F1 |
|------|------------------------------|----------------|
| 1    | Random Forest (default)      | **88.23%**     |
| 2    | Random Forest (grid-tuned)   | 87.98%         |
| 3    | XGBoost                      | 87.89%         |
| 4    | Logistic Regression          | 87.73%         |
| 5    | HistGradientBoosting         | 87.60%         |
| 6    | MLP (deep learning)          | 85.45%         |
| —    | Stratified random baseline   | 8.18%          |

## License

[MIT](./LICENSE)
