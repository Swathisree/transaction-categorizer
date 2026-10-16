"""
Retrain the champion Random Forest on the cleaned dataset and export the
artifacts the API loads at startup (`model.joblib` and `model_meta.json`).

Run from this folder:
    python train_model.py

The input CSV must be at `../transaction-dataset-cleaned.csv` relative to this
file (adjust SRC below if you've moved it).
"""
import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

RNG = 42
HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "transaction-dataset-cleaned.csv"

df = pd.read_csv(SRC, parse_dates=["created_at"])
df["hour"] = df["created_at"].dt.hour
df["dow"]  = df["created_at"].dt.dayofweek
df["pfm_recurring_frequency"] = df["pfm_recurring_frequency"].fillna("none")

NUM = ["amount_value", "fees", "hour", "dow"]
LOW = ["card_name", "digitalization_type", "merchant_pan_entry_method",
       "status", "pfm_recurring_frequency"]
HIG = ["merchant_name"]

y = df["minor_category"]
X = df[NUM + LOW + HIG]

MINOR_TO_MAJOR = (df.drop_duplicates("minor_category")
                    .set_index("minor_category")["major_category"]
                    .to_dict())

pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW),
        ("te",  TargetEncoder(target_type="multiclass", random_state=RNG), HIG),
    ])),
    ("clf", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2,
        random_state=RNG, n_jobs=-1,
    )),
])
print(f"Training champion RF on {len(df):,} rows...")
pipe.fit(X, y)
print("Done.")

joblib.dump(pipe, HERE / "model.joblib", compress=3)
(HERE / "model_meta.json").write_text(json.dumps({
    "model_name":    "Random Forest (tournament champion)",
    "minor_to_major": MINOR_TO_MAJOR,
    "minor_classes":  sorted(y.unique()),
    "major_classes":  sorted(set(MINOR_TO_MAJOR.values())),
    "feature_cols_raw_input": ["amount_value", "fees", "created_at",
                               "card_name", "digitalization_type",
                               "merchant_pan_entry_method", "status",
                               "pfm_recurring_frequency", "merchant_name"],
    "notes": "created_at is parsed to hour and day-of-week internally; "
             "pfm_recurring_frequency null is coerced to the string 'none'; "
             "fees are in euros.",
}, indent=2))
print(f"Wrote {HERE / 'model.joblib'} and {HERE / 'model_meta.json'}")
