"""
Retrain the champion Random Forest on the cleaned dataset and export the
artifacts the API loads at startup (`model.joblib` and `model_meta.json`).

Usage (from this folder):
    python train_model.py                          # default input
    python train_model.py --input /path/to/cleaned.csv --out-dir .
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT   = HERE.parent / "data" / "transaction-dataset-cleaned.csv"
DEFAULT_OUT_DIR = HERE
RNG = 42


def main(input_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(input_csv, parse_dates=["created_at"])
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
    print(f"Training champion RF on {len(df):,} rows (input: {input_csv}) ...")
    pipe.fit(X, y)
    print("Done.")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    meta_path  = out_dir / "model_meta.json"

    joblib.dump(pipe, model_path, compress=3)
    meta_path.write_text(json.dumps({
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
    print(f"Wrote {model_path} and {meta_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the champion RF and emit model.joblib + model_meta.json")
    p.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT,
                   help=f"Path to the cleaned CSV (default: {DEFAULT_INPUT})")
    p.add_argument("--out-dir", "-o", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Directory to write model.joblib + model_meta.json (default: {DEFAULT_OUT_DIR})")
    args = p.parse_args()
    main(args.input, args.out_dir)
